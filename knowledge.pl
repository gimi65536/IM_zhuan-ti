% is_leftist(+Node, +NodeList)
% true if no element AnotherNode in NodeList satisfy edge(AnotherNode, Node)
is_leftist(Node, NodeList) :-
	\+ (member(AnotherNode, NodeList),
		edge(AnotherNode, Node)).

% topological_sort(+NodeList, ?Return)
% Given a node list NodeList, check if Return is a topological sorted form of NodeList
topological_sort(NodeList, [Node | OtherReturn]) :-
	select(Node, NodeList, OtherNodeList),
	is_leftist(Node, OtherNodeList),
	topological_sort(OtherNodeList, OtherReturn).
topological_sort([], []).

grab_word([Index | OtherIndex], [Word | OtherWord]) :-
	word(Index, Word),
	grab_word(OtherIndex, OtherWord).
grab_word([], []).