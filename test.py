import re
import numpy as np
from itertools import combinations, product
from typing import List, Tuple, Any
from collections import namedtuple

AnchorType = np.ndarray
ScoreType = Any #float, int, etc. #Ord Scoretypr #C++ concept: LessThanComparable

class Param():
	__slot__ = ('_init_value', '_base_point', '_merge_point')
	def __init__(self, init_value = None, base_point = None, merge_point = None):
		self._init_value = init_value
		self._base_point = base_point
		self._merge_point = merge_point
	@property
	def init_value(self):
		return self._init_value
	@init_value.setter
	def init_value(self, n):
		self._init_value = n
	@property
	def base_point(self):
		return self._base_point
	@base_point.setter
	def base_point(self, f):
		#f :: List[str] -> List[str] -> float
		self._base_point = f
	@property
	def merge_point(self):
		return self._merge_point
	@merge_point.setter
	def merge_point(self, f):
		#f :: (List[str], List[str]) -> (List[str], List[str]) -> (float, List[anchor]) -> (float, List[anchor]) -> float
		self._merge_point = f
	

class StringAlign():
	c1, c2, c3 = re.compile(R"([^\w\s'])"), re.compile(R"\s+"), re.compile(R"^\s|\s$")
	init_similarity = 0
	State = namedtuple("State", "length Dict")
	join = staticmethod(lambda l: " ".join(l))
	def __init__(self, *args):
		self._state = None
		if len(args) > 0 and type(args[0]) is type(self): #copy and push
			self._l = args[0]._l[:]
			self.push(*args[1:])
		else:
			self._l = []
			self.push(*args)
	def push(self, *args):
		l = []
		for i in args:
			if type(i) is str:
				l.append(i)
			else:
				l.extend(i)
		l = [self.c3.sub("", self.c2.sub(" ", self.c1.sub(R" \1 ", s))).split() for s in l]
		self._l.extend(l)
	def push_list(self, l):
		self.push(l)
	def concat(self, n):
		self._l.extend(n._l)
	def __iadd__(self, n):
		if type(n) is not type(self):
			self.push_list(n)
		else:
			self.concat(n)
		return self
	def __add__(self, n):
		sol = type(self)(self) #create a copy
		sol.__iadd__(n)
		return sol
	def __radd__(self, n):
		return self.__add__(n)
	def __str__(self):
		state, l, join = self._state, self._l, self.join
		if state is None:
			return "No state is ready."
		n = state.length
		s = ""
		for i, j in combinations(range(n), 2):
			ans = state.Dict[(i, j)]
			s += "string {}\n{}\n{}\nhas similarity {}\nfix points are: {}\n\n".format((i, j), join(l[i]), join(l[j]), ans[0], [tuple(i) for i in ans[1]])
		return s[:-2]
	def evaluate(self, param : Param):
		l = self._l
		n = len(l)
		state = self.__class__.State(n, dict())
		for i, j in combinations(range(n), 2):
			get = self.__class__.compare(l[i], l[j], param)
			get = get[0], list(get[1])
			state.Dict[(i, j)] = get
		self._state = state
	@classmethod
	def compare(cls, l1 : List[str], l2 : List[str], param : Param):
		anchors = cls._anchors(l1, l2)
		return cls._compare_detail(l1, l2, param, anchors)
	@staticmethod
	def _anchors(l1 : List[str], l2 : List[str]) -> List[AnchorType]:
		s = set(l1)
		s &= set(l2)
		sol = []
		for c in s:
			i1, i2 = np.argwhere(np.array(l1) == c).reshape([-1]), np.argwhere(np.array(l2) == c).reshape([-1])
			sol.extend([np.array(i) for i in product(i1, i2)])
		return sol
	@classmethod
	def _compare_split(cls, l1 : List[str], l2 : List[str], param : Param, anchors : List[AnchorType], now_anchor : AnchorType) -> Tuple[ScoreType, List[AnchorType]]:
		#return (float, List[anchor]) as the highest similarity, anchors
		if now_anchor is None: #base case
			return param.base_point(l1, l2), []
		#print(now_anchor)
		left_child = (l1[:now_anchor[0]], l2[:now_anchor[1]])
		right_child = (l1[(now_anchor[0] + 1):], l2[(now_anchor[1] + 1):])
		left_anchor = [anchor for anchor in anchors if np.all(anchor < now_anchor)]
		right_anchor = [anchor - now_anchor - 1 for anchor in anchors if np.all(anchor > now_anchor)]
		left_ans = cls._compare_detail(*left_child, param, left_anchor)
		right_ans = cls._compare_detail(*right_child, param, right_anchor)
		sol_anchor = left_ans[1] + [now_anchor] + [anchor + now_anchor + 1 for anchor in right_ans[1]]
		sol_simi = param.merge_point(left_child, right_child, left_ans, right_ans)
		return sol_simi, sol_anchor
	@classmethod
	def _compare_detail(cls, l1 : List[str], l2 : List[str], param : Param, anchors : List[AnchorType]) -> Tuple[ScoreType, List[AnchorType]]:
		similarity, anchors_to_choose = param.init_value, []
		if len(anchors) == 0:
			simi, use_anchors = cls._compare_split(l1, l2, param, anchors, None) #call base case
			if simi > similarity:
				similarity, anchors_to_choose = simi, use_anchors
		for anchor in anchors: #if executing above, the for-loop will not be executed
			#print(anchor)
			simi, use_anchors = cls._compare_split(l1, l2, param, anchors, np.array(anchor))
			if simi > similarity:
				similarity, anchors_to_choose = simi, use_anchors
		#print('{}\n{}\n{}, {}\n\n'.format(l1, l2, similarity, anchors_to_choose))
		return similarity, anchors_to_choose

p = Param()
way = 'james'
if way == 'wayne':
	p.init_value = 0.0
	p.base_point = lambda l1, l2: 1 / (len(l1) + len(l2) + 1)
	def merge_point(left, right, ans1, ans2):
		left_len = len(left[0]) + len(left[1])
		right_len = len(right[0]) + len(right[1])
		left_point, right_point = ans1[0], ans2[0]
		return (left_point * left_len + right_point * right_len + 2) / (left_len + right_len + 2)
	p.merge_point = merge_point
elif way == 'james':
	p.init_value = -np.inf
	p.base_point = lambda l1, l2: -(len(l1) + len(l2))
	p.merge_point = lambda l, r, a1, a2: a1[0] + a2[0] + 2

S = StringAlign()
S += ['a b c a d', 'b a d', 'a c a d a', 'a b a']

S.evaluate(p)
print(S)