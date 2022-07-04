from relevanceai.operations_new.concept_graph.transform import ConceptGraphTransform
import pickle
from sklearn.datasets import fetch_20newsgroups


if __name__ == '__main__':
    X, y = fetch_20newsgroups(subset = 'test', shuffle = False, remove=('headers', 'footers', 'quotes'), return_X_y = True)
    data = []
    for n, sent in enumerate(X[:1000]):
        doc = " ".join(sent.replace('\n', ' ').split())
        data.append({'content':doc})

    # print(data[9]['content'])



    model = ConceptGraphTransform(max_number_of_clusters = 15, min_number_of_clusters = 3, number_of_concepts = 100)
    graph = model.transform(data)

    print(graph)