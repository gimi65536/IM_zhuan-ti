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
	def push(self, *args):
		for i in args:
			if i in self._dict:
				continue
			self._dict[i] = _node()
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
	def sets(self):
		d = self._dict
		dic = dict()
		for key, node in d.items():
			ptr = node.find_root()
			if id(ptr) not in dic:
				dic[id(ptr)] = []
			dic[id(ptr)].append(key)
		return dic.values()
	def __str__(self):
		d = self.sets()
		s = [f"{{{', '.join([str(i) for i in l])}}}" for l in d]
		return f"{{{', '.join(s)}}}"

if __name__ == '__main__':
	#test
	x = disjoint_set(*range(10))
	x.union(0, 2)
	print(x.is_same(0, 2), x.is_same(0, 3))
	print(x)
	x.union(2, 3)
	print(x.is_same(0, 2), x.is_same(0, 3))
	print(x)