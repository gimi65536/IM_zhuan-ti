class _node():
	def __init__(self):
		self.ptr = None
		self.rank = 0
	def concat(self, ptr):
		self.ptr = ptr
	def find_root(self):
		ptr = self
		while True:
			if ptr.ptr is None:
				return ptr
			ptr = ptr.ptr

class disjoint_set():
	def __init__(self, *args):
		self._dict = dict()
		self.push(*args)
	@classmethod
	def from_iterable(cls, l):
		return cls(*l)
	def push(self, *args):
		for i in args:
			if i in self._dict:
				continue
			self._dict[i] = _node()
	def push_iterable(self, l):
		self.push(*l)
	def find(self, n):
		if n not in self._dict:
			return None
		return self._dict[n].find_root()
	def is_same(self, m, n) -> bool:
		return self.find(m) is self.find(n)
	def union(self, m, n):
		i, j = self.find(m), self.find(n)
		if i is None or j is None:
			return
		if i is j:
			return
		if i.rank > j.rank:
			j.concat(i)
		else:
			i.concat(j)
			if i.rank == j.rank:
				j.rank += 1
	def index(self): #many to one
		return {key: id(node.find_root()) for key, node in self._dict.items()}
	def reversed_index(self): #one to many, so using list
		index = self.index()
		dic = dict()
		for key, i in index.items():
			if i not in dic:
				dic[i] = []
			dic[i].append(key)
		return dic
	def sets(self):
		dic = self.reversed_index()
		return dic.values()
	def __str__(self):
		d = self.sets()
		s = [f"{{{', '.join([str(i) for i in l])}}}" for l in d]
		return f"{{{', '.join(s)}}}"
	def rebuild(self):
		d = self.sets()
		for l in d:
			if len(l) == 1:
				self._dict[l[0]] = _node()
			else:
				n = _node()
				n.rank = 1
				self._dict[l[0]] = n
				for k in l[1:]:
					p = _node()
					p.concat(n)
					self._dict[k] = p

if __name__ == '__main__':
	#test
	x = disjoint_set(*range(10))
	x.union(0, 2)
	print(x.is_same(0, 2), x.is_same(0, 3))
	print(x)
	x.union(2, 3)
	print(x.is_same(0, 2), x.is_same(0, 3))
	print(x)