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


class NoColonError(Exception):
    pass


class NoEntity(Exception):
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


def is_entity(z: AMRNode):
    return z.ner is not None


class AMRGraph:
    def __init__(self, lines):
        self.nodes = []  # list of AMRNode objects
        self.adjacency = []  # The node at nodes[i] has the sons listed at adjacency[i]: list of AMRLink objects
        self.sentence = lines[3][8:]
        self.current_leaf_idx = 0  # we create nodes for leaves such as ':polarity -' or ':month 6'
        self.original_text_representation = lines.copy()
        self._parse_lines(lines[4:])
        self._find_word_intervals()

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

            # at this point parent_idx IS the parent index
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
