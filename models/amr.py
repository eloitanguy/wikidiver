import torch
import os

from models.utils import find_most_similar_word_idx_interval, HiddenPrints, length_of_longest_common_subsequence
import multiprocessing as mp
from multiprocessing.dummy import Pool
from config import AMR_CONFIG
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # disable annoying TF logs (TF is loader by transformers systematically)

from models.spring_amr.penman import encode
from models.spring_amr.utils import instantiate_model_and_tokenizer

OPS = ['op1', 'op2', 'op3', 'op4', 'op5']

ARGS_NONZERO = ['ARG1', 'ARG2', 'ARG3', ':ARG4', 'ARG5', 'ARG6', 'ARG7', 'ARG8', 'ARG9', 'ARG10', 'ARG11', 'ARG12',
                'ARG13', 'ARG14', 'ARG15', 'ARG16', 'ARG17', 'ARG18', 'ARG19', 'ARG20']

# exceptions to the rule "a description indicates a verb iif it has a '-'."
NOT_VERBS = ['date-entity', 'government-organization', 'temporal-quantity', 'amr-unknown', 'multi-sentence',
             'political-party', 'monetary-quantity', 'ordinal-entity', 'religious-group', 'percentage-entity',
             'world-region', 'url-entity', 'political-movement', 'et-cetera', 'at-least', 'mass-quantity']


class NoColonError(Exception):
    pass


class NoEntity(Exception):
    pass


class NoPath(Exception):
    pass


class AMRNode:
    def __init__(self, ID, description, name=None, wiki=None, ner=None, start_idx=-1, end_idx=-1):
        self.id = ID
        self.description = description
        self.name = name
        self.wiki = wiki
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.ner = ner

    def update(self, name=None, wiki=None, ner=None, start_idx=-1, end_idx=-1):
        if name:
            self.name = name
        if wiki:
            self.wiki = wiki
        if start_idx != -1:
            self.start_idx = start_idx
        if end_idx != -1:
            self.end_idx = end_idx
        if ner:
            self.ner = ner

    def __str__(self):
        s = 'AMRVariable {}:\tdescription: {}'.format(self.id, self.description)
        if self.name:
            s = s + '\n\tname: ' + self.name
        if self.wiki:
            s = s + '\n\twiki name: ' + self.wiki
        if self.start_idx != -1 and self.end_idx != -1:
            s = s + '\n\tbounds: ({}, {})'.format(self.start_idx, self.end_idx)
        if self.ner:
            s = s + '\n\tNER: name={}, mention={}, ID={}'.format(self.ner['name'], self.ner['mention'], self.ner['id'])

        return s

    def __repr__(self):
        return self.__str__()


class AMRLink:
    def __init__(self, op, to_node_id, to_node_idx):
        self.op = op
        self.to_node_id = to_node_id
        self.to_node_idx = to_node_idx

    def __str__(self):
        return 'AMRLink\top: {}\tto_node_id: {}\tto_node_idx: {}'.format(
            self.op, self.to_node_id, self.to_node_idx
        )

    def __repr__(self):
        return self.__str__()


def is_entity(z: AMRNode, g):
    if z.ner is not None:
        return True

    # We must check whether z is a leaf referring to an entity variable
    if z.id[0] == 'l':  # all leaves start with l
        if z.description in g.node_id_to_idx:  # in this case z is indeed a variable leaf
            return g.nodes[g.node_id_to_idx[z.description]].ner is not None  # check if the associated var is an entity

    return False


def refers_to(leaf: AMRNode, z: AMRNode):
    """
    Returns True iff l is a leaf referring to the variable z in their AMR graph.
    """
    return leaf.id[0] == 'l' and leaf.description == z.id


def is_verb(z: AMRNode):
    return '-' in z.description and z.description not in NOT_VERBS


class AMRGraph:
    def __init__(self, lines):
        self.nodes = []  # list of AMRNode objects
        self.adjacency = []  # The node at nodes[i] has the sons listed at adjacency[i]: list of AMRLink objects
        self.sentence = lines[3][8:]
        self.current_leaf_idx = 0  # we create nodes for leaves such as ':polarity -' or ':month 6'
        self.original_text_representation = lines.copy()
        self._parse_lines(lines[4:])
        self._find_word_intervals()
        self.node_id_to_idx = {node.id: idx for (idx, node) in enumerate(self.nodes)}

    def _parse_line(self, line):
        line = line.replace(')', '').replace('"', '').replace('\n', '')
        colon_idx = line.find(':')

        # no indentation, should only happen on the first line which is handled outside this function
        if colon_idx == -1:
            print('Found no colon in the input line: ', line)
            raise NoColonError

        line_elements = line[colon_idx + 1:].split(' ')
        op = line_elements[0]

        if line_elements[1].find('(z') != -1:  # in this case we have a variable (eg z12)
            z = line_elements[1][1:]
            assert line_elements[2] == '/', "the 3rd word in the line {} is not '/'".format(line)
            description = line_elements[3]
            node = AMRNode(ID=z, description=description)

        else:  # in this case we have a leaf operator such as ':polarity -' or ':year 17'
            node = AMRNode(ID='l' + str(self.current_leaf_idx), description=' '.join(line_elements[1:]))
            op = line_elements[0]
            self.current_leaf_idx += 1

        return colon_idx, op, node

    def _parse_lines(self, lines):
        # we start by parsing the first line individually (its format is unique)
        first_line_elements = lines[0].split(' ')
        z0 = AMRNode(ID=first_line_elements[0][1:], description=first_line_elements[2].replace('\n', ''))

        # we parse each line into (indentation number, operator, AMRNode object)
        parsed_lines = [(0, None, z0)] + [self._parse_line(line) for line in lines[1:]]

        # we now find the parent of each parsed node using the indentation numbers and add them to the tree
        self.nodes.append(z0)
        self.adjacency.append([])  # preparing the list of children of the new node
        line_idx_to_node_idx = {0: 0}  # the first line refers to the first node in self.nodes

        # here the indices of lines containing the words in a name will be stored for skipping,
        # since we only add the name to the variable's description instead of adding a 'name' node
        parsing_indices_to_skip = []

        for parsing_idx in range(1, len(parsed_lines)):

            if parsing_idx in parsing_indices_to_skip:
                continue

            # the parent of the current node is the first previous node that has a smaller indent
            parent_idx = parsing_idx - 1
            current_indent = parsed_lines[parsing_idx][0]

            while parsed_lines[parent_idx][0] >= current_indent:  # we look for the parent index
                parent_idx -= 1

            # at this point 'parent_idx' IS the parent index
            op, child_node = parsed_lines[parsing_idx][1], parsed_lines[parsing_idx][2]

            # if op in ['name', 'wiki'], the child node is just information on the parent node:
            # no new node, just an update to the parent

            if op == 'name':  # updating the parent's name: we have a node 'name' with a child for each word
                name_indent = parsed_lines[parsing_idx + 1][0]  # the indent of the children of 'name' (words)
                potential_word_index = parsing_idx + 1
                name_list = []

                # going through the next lines: if they have the same indent as the line following 'name', then they
                # are of the form :op[n] "word" where [n] is a number and "word" is a word from the name.
                # We note these lines as future skips and we add the words to the name.
                while potential_word_index < len(parsed_lines) and parsed_lines[potential_word_index][0] == name_indent:
                    parsing_indices_to_skip.append(potential_word_index)
                    name_list.append(parsed_lines[potential_word_index][2].description)
                    potential_word_index += 1

                self.nodes[line_idx_to_node_idx[parent_idx]].update(name=' '.join(name_list))  # updating the parent

            elif op == 'wiki':  # updating the parent's wiki title name (the line looks like :wiki 'Wiki_name')
                self.nodes[line_idx_to_node_idx[parent_idx]].update(wiki=child_node.description)

            else:  # we have a usual node to add to the tree
                self.nodes.append(child_node)
                self.adjacency.append([])  # preparing the list of children of the new node
                child_node_id, child_node_idx = child_node.id, len(self.nodes) - 1
                line_idx_to_node_idx[parsing_idx] = child_node_idx  # updating the line - to - node list dict

                # adding the link to the list of children of the parent in the adjacency list
                self.adjacency[line_idx_to_node_idx[parent_idx]].append(AMRLink(op, child_node_id, child_node_idx))

    def _find_word_intervals(self):
        for var in self.nodes:
            if var.name:
                name = var.name
            elif var.wiki:
                name = var.wiki
            else:
                name = var.description
            var.start_idx, var.end_idx = find_most_similar_word_idx_interval(self.sentence, name)

    def ner_results(self):
        return [var.ner for var in self.nodes if var.ner]

    def __str__(self):
        s = 'AMRGraph for sentence "{}"\n----NODES----\n'.format(self.sentence.replace('\n', ''))

        for n in self.nodes:
            s = s + str(n) + '\n'

        s = s + '----ADJACENCY----\n'

        for parent_idx, list_of_children in enumerate(self.adjacency):
            if list_of_children:
                s = s + self.nodes[parent_idx].id + '\n'
                for child_link in list_of_children:
                    s = s + '  ---> ' + str(child_link) + '\n'
        return s

    def __repr__(self):
        return self.__str__()

    def to_tikz(self):
        def tikz_node_type(node):
            if is_verb(node):
                node_type = 'v'
            elif node.ner:
                node_type = 'e'
            else:
                node_type = 'n'
            return node_type

        root = self.nodes[0]
        tikz_string_list = [
            '\\begin{tikzpicture}',
            '\t[',
            '\t\tv/.style={rectangle, draw=red!60, fill=red!5, very thick, minimum size=5mm},',
            '\t\tn/.style={rectangle, draw=blue!60, fill=blue!5, very thick, minimum size=5mm},',
            '\t\tz/.style={rectangle, draw=green!60, fill=green!5, very thick, minimum size=5mm},',
            '\t\tlevel distance=6em,',
            '\t\tlevel 1/.style={sibling distance=10em},  % TO EDIT',
            '\t\tlevel 2/.style={sibling distance=15em},  % TO EDIT',
            '\t\tlevel 3/.style = {sibling distance = 8em}, % TO EDIT',
            '\t\tlevel 4/.style = {sibling distance = 8em}, % TO EDIT',
            '\t\tlevel 5/.style = {sibling distance = 8em}, % TO EDIT',
            '\t\tlevel 6/.style = {sibling distance = 8em}, % TO EDIT',
            '\t\tlevel 7/.style = {sibling distance = 8em}, % TO EDIT',
            '\t\tlevel 8/.style = {sibling distance = 8em}, % TO EDIT',
            '\t\tlevel 9/.style = {sibling distance = 8em}, % TO EDIT',
            '\t\tlevel 10/.style = {sibling distance = 8em}, % TO EDIT',
            '\t\tsloped',
            '\t]',
            '\t\\node[{}] {{{} / {}}}'.format(tikz_node_type(root), root.id, root.description)
        ]

        def to_tikz_from_node_idx(node_idx, current_tab, op_to_this_node):
            children = self.adjacency[node_idx]
            node = self.nodes[node_idx]
            node_type = tikz_node_type(node)
            tab = '\t' * current_tab
            if 'l' not in node.id:
                res = [tab + 'child {{ node [{}] {{{} / {}}}'.format(node_type, node.id, node.description)]
            else:
                res = [tab + 'child {{ node [{}] {{{}}}'.format(node_type, node.description)]

            for child in children:
                res.extend(to_tikz_from_node_idx(child.to_node_idx, current_tab + 1, child.op))

            res = res + ['\t' * (current_tab + 1) + 'edge from parent node [above] {{:{}}} }}'.format(op_to_this_node)]
            return res

        for link in self.adjacency[0]:
            tikz_string_list.extend(to_tikz_from_node_idx(link.to_node_idx, 2, link.op))

        tikz_string_list[-1] = tikz_string_list[-1] + ';'  # concluding the tikz env with a semicolon
        tikz_string_list.append('\end{tikzpicture}')

        return '\n'.join(tikz_string_list)


def find_path_between(start_node_idx, end_node_idx, g: AMRGraph):
    """
    Returns a path from start_node_idx of the AMR tree g to end_node_idx. \n
    Requires start_node_idx to be higher in the tree!\n
    The only rule from suggest_entity_pairs that we follow is about the polarity, the others are checked later.\n
    :return: the path: list of [AMRNode, AMRLink], raise NoPath if there is no path.
    """
    subtree_root = g.nodes[start_node_idx]
    if start_node_idx == end_node_idx:  # Checking if the two nodes are the same already
        # the link empty link that should not be used to ensure correct typing
        return [[subtree_root, AMRLink(op='-1', to_node_id='-1', to_node_idx=-1)]]
    for link in g.adjacency[start_node_idx]:  # if there are no children, we have already checked if they are the same
        if link.op == 'polarity':  # no negative facts so we forbid negative propositions altogether.
            raise NoPath

        son_id = link.to_node_id
        son_idx = g.node_id_to_idx[son_id]

        try:
            path_from_son_to_end = find_path_between(son_idx, end_node_idx, g)
        except NoPath:
            continue

        # if we haven't encountered NoPath and skipped with 'continue' then we output our found path
        return [[subtree_root, link]] + path_from_son_to_end

    # if we have exhausted the sons and not returned anything then there is not path
    raise NoPath


def find_lowest_common_ancestor(e1_path_from_root, e2_path_from_root):
    """finds the lowest common ancestor (LCA) by going down the two paths, helper to suggest_entity_pairs.\n
    takes as input two lists of (AMRNode, AMRLink) tuples."""
    for ancestor_idx in range(min(len(e1_path_from_root), len(e2_path_from_root))):
        # the first difference is just under the LCA
        if e1_path_from_root[ancestor_idx][0].id != e2_path_from_root[ancestor_idx][0].id:
            return ancestor_idx - 1

    # if we're at the end then either the last node is the LCA (and one path is included in another)
    return min(len(e1_path_from_root), len(e2_path_from_root)) - 1


def suggest_entity_pairs(g: AMRGraph):
    nodes, adjacency = g.nodes, g.adjacency
    entities = [node for node in nodes if is_entity(node, g)]
    n_entities = len(entities)

    # We go through the nodes one by one: we look at their children (at adjacency[node_idx]), which are under the node
    # where the variable behind the node is defined (i.e. first seen in the AMR tree).

    # We compute the 'distances' between each entity pair by saving the path of each entity to the root, then finding
    # the common sub-paths for each entity pair.
    # The 'distance' follows the following rules:
    #   - if there is no verb in the path, d=Inf
    #   - if the two entities are :ARG0 and :ARGn (n > 0) of the same verb, then d=inf, same for :op s. (manual check)
    #   - if one of the nodes on the path has a :polarity - attribute, then d=inf

    # Once we have all the distances, the proposed associate for each entity its closest entity (if any).

    # Remark on multiple references of the same AMR variable:
    # A link z_n (current node) -> z_m (other node) where z_m has already been defined is encoded in our AMR Graph
    # as z_n -> l_k where l_k is a leaf with a description='z_m'. This makes our structure a rigorous tree.

    # --- STEP 1 --- compute the paths to the AMR tree root (always z0 of idx 0)
    paths_from_root = []
    for node in entities:
        try:
            paths_from_root.append(find_path_between(0, g.node_id_to_idx[node.id], g))
        except NoPath:
            paths_from_root.append(None)

    # --- STEP 2 --- compute the paths between the pairs using the lowest common ancestor by intersecting the root paths
    # We also check the constraints on the path, leaving it as 9999 if invalid and compute the pair distances
    pair_distances = [[9999] * n_entities for _ in range(n_entities)]  # 9999 ~ Inf
    for e1_idx in range(n_entities):
        for e2_idx in range(e1_idx + 1, n_entities):
            e1_path_from_root, e2_path_from_root = paths_from_root[e1_idx], paths_from_root[e2_idx]

            if e1_path_from_root is None or e2_path_from_root is None:  # if there is already no path, continue
                continue

            LCA_idx_in_path_from_root = find_lowest_common_ancestor(e1_path_from_root, e2_path_from_root)
            # path e1 -> ... > LCA (flipped the node order but the links are broken)
            path_e1_to_lca = e1_path_from_root[:LCA_idx_in_path_from_root - 1:-1] if LCA_idx_in_path_from_root != 0 \
                else e1_path_from_root[::-1]
            # path lCA -> [node -> ... -> e2] (LCA is excluded)
            path_lca_excluded_to_e2 = e2_path_from_root[LCA_idx_in_path_from_root + 1:]

            # flipping back the link, right now they are from i+1 to i
            for idx in range(len(path_e1_to_lca) - 1):
                link_from_next_to_here = path_e1_to_lca[idx + 1][1]
                next_node = path_e1_to_lca[idx + 1][0]
                link_from_here_to_next = AMRLink(op=link_from_next_to_here.op,
                                                 to_node_id=next_node.id,
                                                 to_node_idx=g.node_id_to_idx[next_node.id])
                path_e1_to_lca[idx][1] = link_from_here_to_next  # updating the link to the right direction

            # writing the link from the LCA to the next node in the path lca -> ... -> e2 (downward)
            path_e1_to_lca[-1][1] = e1_path_from_root[LCA_idx_in_path_from_root][1]
            path = path_e1_to_lca + path_lca_excluded_to_e2
            LCA_idx_in_e1_to_e2_path = len(path_e1_to_lca) - 1  # the lca is the last element of this list

            # we check whether the LCA in the path has invalid operators: this is the case if we have:
            #           op n                 op m
            # node ------up-------> LCA -----down-----> node      (first 'if')
            # Or if we have the same with ARGn, ARGm, n,m>0 when there is also an :ARG0     (second 'if')
            # for a path to be able to take this test it needs to have a node before the LCA, ie lca_idx > 0

            if LCA_idx_in_e1_to_e2_path > 0:
                if path[LCA_idx_in_path_from_root - 1][1].op in OPS and path[LCA_idx_in_path_from_root][1].op in OPS:
                    continue  # we leave the distance at 9999 (Inf basically)

                lca_node_idx = g.node_id_to_idx[e1_path_from_root[LCA_idx_in_path_from_root][0].id]  # lca idx @ g.nodes
                if 'ARG0' in [link.op for link in g.adjacency[lca_node_idx]] and LCA_idx_in_e1_to_e2_path != 0:
                    if path[LCA_idx_in_e1_to_e2_path - 1][1].op in ARGS_NONZERO \
                            and path[LCA_idx_in_e1_to_e2_path][1].op in ARGS_NONZERO:
                        continue  # we leave the distance at 9999

            # we check if there is a verb in the path: if there isn't, the path is invalid.
            found_verb = False
            for path_tuple in path:
                if is_verb(path_tuple[0]):
                    found_verb = True

            if not found_verb:
                continue  # we leave the distance at 9999

            pair_distances[e1_idx][e2_idx] = len(path) - 2  # the path includes the endings
    # --- STEP 3 --- update the distances so that if a leaf l referring to an entity e1 is close to e2,
    # then e1 is saved as close to e2. First we find the references, then we update the distances.
    referring_to_list = [{'id': '-1', 'idx': -1} for _ in range(n_entities)]
    for e1_idx, e1 in enumerate(entities):
        if 'z' in e1.id:  # this is not a leaf variable
            referring_to_list[e1_idx] = {'id': e1.id, 'idx': e1_idx}

        else:  # then this could be a leaf variable
            found_reference = False
            for e2_idx, e2 in enumerate(entities):  # we check for other variables that e1 could refer to
                if refers_to(e1, e2) and e1.id != e2.id:
                    referring_to_list[e1_idx] = {'id': e2.id, 'idx': e2_idx}
                    found_reference = True

            # if found_reference == False, it was actually not a leaf variable
            if not found_reference:
                referring_to_list[e1_idx] = {'id': e1.id, 'idx': e1_idx}

    for e1_idx in range(n_entities):
        for e2_idx in range(e1_idx + 1, n_entities):
            # the distance between the two diff. entities is the closest distance between any of their representatives
            if referring_to_list[e1_idx]['idx'] != referring_to_list[e2_idx]['idx']:
                pair_distances[referring_to_list[e1_idx]['idx']][referring_to_list[e2_idx]['idx']] = \
                    min(pair_distances[e1_idx][e2_idx],
                        pair_distances[referring_to_list[e1_idx]['idx']][e2_idx],
                        pair_distances[e1_idx][referring_to_list[e2_idx]['idx']],
                        pair_distances[referring_to_list[e1_idx]['idx']][referring_to_list[e2_idx]['idx']])

    # --- STEP 4 --- for each entity, proposes the closest legal entities (list if equal distance)
    suggested_pair_ids = []
    for e1_idx in range(n_entities):

        # only find pairs between non-leaves -> no pair repetition
        if referring_to_list[e1_idx]['id'] != entities[e1_idx].id:
            continue

        best_distance = 9999
        best_e2_indices = []
        for e2_idx in range(e1_idx + 1, n_entities):

            # only find pairs between non-leaves -> no pair repetition
            if referring_to_list[e2_idx]['id'] != entities[e2_idx].id:
                continue

            if pair_distances[e1_idx][e2_idx] < best_distance:  # if we beat the record, we start a new list
                best_distance = pair_distances[e1_idx][e2_idx]
                best_e2_indices = [e2_idx]

            elif pair_distances[e1_idx][e2_idx] == best_distance:  # same distance -> add to the podium list
                best_e2_indices.append(e2_idx)

        if best_distance != 9999:  # if there are legal propositions, give them
            e1_id = referring_to_list[e1_idx]['id']
            for best_e2_idx in best_e2_indices:
                e2_id = referring_to_list[best_e2_idx]['id']
                suggested_pair_ids.append({'e1': e1_id, 'e2': e2_id, 'd': best_distance})

    return suggested_pair_ids


class AMRParser:
    """
    Uses the SPRING AMR parsing model to parse a raw text sentence into an AMRGraph.
    Code adapted from https://github.com/SapienzaNLP/spring/blob/main/bin/predict_amrs_from_plaintext.py
    """

    def __init__(self, config=AMR_CONFIG):
        with HiddenPrints():  # avoid TF logs
            model_name = 'facebook/bart-large'
            self.model, self.tokenizer = instantiate_model_and_tokenizer(
                model_name,
                dropout=0.,
                attention_dropout=0,
                penman_linearization=True,
                use_pointer_tokens=True,
            )
            self.model.load_state_dict(torch.load('models/spring_amr/AMR3.pt', map_location='cpu')['model'])
            self.device = torch.device('cuda')
            self.model.to(self.device)
            self.model.eval()
            self.model.amr_mode = True
            self.beam_size = config['beam_size']
            self.entity_threshold = config['entity_threshold']
            self.n_aliases = config['n_aliases']

            with open('wikidatavitals/data/entity_aliases.json', 'r') as f:
                self.entity_aliases = json.load(f)

    @staticmethod
    def _try_alias(snippet, alias):
        return len(snippet) < 3 * len(alias) and len(alias) < 3 * len(snippet)  # none 3x bigger

    def _find_entity(self, snippet):
        best_LCS = -1
        length_of_best = 99999

        for entity_id, aliases in self.entity_aliases.items():
            for alias in aliases[:self.n_aliases]:
                if self._try_alias(snippet, alias):
                    LCS = length_of_longest_common_subsequence(snippet, alias)

                    if LCS > best_LCS or (LCS == best_LCS and len(alias) < length_of_best):
                        best_LCS, best_id, length_of_best = LCS, entity_id, len(alias)
                        if best_LCS / max(length_of_best, len(snippet)) > self.entity_threshold:
                            return best_id

        raise NoEntity

    def _get_entity_from_var(self, var, sentence):
        if var.start_idx == -1 or var.end_idx == -1:  # should never happen but in that case the var is unusable
            return {}
        if var.name:
            name = var.name
        elif var.wiki:
            name = var.wiki
        else:
            name = var.description
            if '-' in name:  # an AMR description with a '-' is a verb and thus unlikely to be an entity -> skip
                return {}
            if len(name) < 4:  # too short: could be an acronym of a pronoun -> skip
                return {}

        try:
            entity_id = self._find_entity(name)
        except NoEntity:
            return {}

        return {
            'start_idx': var.start_idx,
            'end_idx': var.end_idx,
            'id': entity_id,
            'name': self.entity_aliases[entity_id][0],  # the first alias is the Wikidata entity name
            'mention': ' '.join(sentence.split(' ')[var.start_idx:var.end_idx + 1])
        }

    def parse_text(self, sentence, NER=True):
        x, _ = self.tokenizer.batch_encode_sentences((sentence,), device=self.device)
        out = self.model.generate(**x, max_length=512, decoder_start_token_id=0, num_beams=self.beam_size)
        graph, status, _ = self.tokenizer.decode_amr(out[0].tolist(), restore_name_ops=True)
        graph.metadata['status'] = str(status)
        graph.metadata['source'] = 'NA'
        graph.metadata['nsent'] = 'NA'
        graph.metadata['snt'] = sentence
        penman_string = encode(graph)
        g = AMRGraph(penman_string.split('\n'))

        if NER:
            workers = mp.cpu_count()
            pool = Pool(workers)
            results_by_var = pool.starmap(self._get_entity_from_var, [(var, sentence) for var in g.nodes])

            for var_idx, var in enumerate(g.nodes):  # update node information
                ner = results_by_var[var_idx]
                if ner != {}:
                    var.update(ner=ner)
            pool.close()

        return g
