class NoColonError(Exception):
    pass


class AMRNode:
    def __init__(self, ID, description, name=None, wiki=None, start_idx=None, end_idx=None):
        self.id = ID
        self.description = description
        self.name = name
        self.wiki = wiki
        self.start_idx = start_idx
        self.end_idx = end_idx

    def update(self, name=None, wiki=None, start_idx=None, end_idx=None):
        if name:
            self.name = name
        if wiki:
            self.wiki = wiki
        if start_idx:
            self.start_idx = start_idx
        if end_idx:
            self.end_idx = end_idx

    def __str__(self):
        s = 'AMRVariable {}:\tdescription: {}'.format(self.id, self.description)
        if self.name:
            s = s + '\tname: ' + self.name
        if self.wiki:
            s = s + '\twiki name: ' + self.wiki
        if self.start_idx and self.end_idx:
            s = s + '\tbounds: ({}, {})'.format(self.start_idx, self.end_idx)

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


class AMRTree:
    def __init__(self, lines):
        self.nodes = []  # list of AMRNode objects
        self.adjacency = []  # The node at nodes[i] has the sons listed at adjacency[i]: list of AMRLink objects
        self.sentence = lines[3][8:]
        self.current_leaf_idx = 0  # we create nodes for leaves such as ':polarity -' or ':month 6'
        self.original_text_representation = lines.copy()

        self._parse_lines(lines[4:])

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
                while parsed_lines[potential_word_index][0] == name_indent:
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

    def __str__(self):
        s = 'AMRTree for sentence "{}"\n----NODES----\n'.format(self.sentence.replace('\n', ''))

        for n in self.nodes:
            s = s + str(n) + '\n'

        s = s + '----ADJACENCY----\n'

        for parent_idx, list_of_children in enumerate(self.adjacency):
            if list_of_children:
                s = s + self.nodes[parent_idx].id + '->\n'
                for child_link in list_of_children:
                    s = s + '\t' + str(child_link) + '\n'
        return s

    def __repr__(self):
        return self.__str__()
