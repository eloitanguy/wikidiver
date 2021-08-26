from models.amr import get_most_recurrent_sub_path, AMRParser, get_simplified_path_list


sentences = []
pair_node_ids = []
relation_name = input('Enter the relation name: ')
relation_id = input('Enter the relation Wikidata ID: ')
print('Setting up the AMR Parser ...')
amr_parser = AMRParser()
print('Starting the input sentence, loop: end by typing STOP')

while True:
	sent = input('Enter a sentence reflecting {} or type STOP: '.format(relation_name))
	if sent == 'STOP':
		break
	print('Parsing the sentence to AMR...')
	g = amr_parser.parse_text(sent, NER=False)
	for line in g.original_text_representation[4:]:
		print(line)
	e1_node_id = input('Enter the node ID for e1 (the order does matter): ')
	if e1_node_id == 'STOP':
		break
	e2_node_id = input('Enter the node ID for e2 (the order does matter): ')
	if e2_node_id == 'STOP':
		break
	sentences.append(sent)
	pair_node_ids.append([e1_node_id, e2_node_id])

print('Ended input sentence loop, computing the most common sub-path ...')

sub_path = get_most_recurrent_sub_path(sentences, pair_node_ids)

print(sub_path)
