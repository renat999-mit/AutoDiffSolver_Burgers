class Surreal:
	"""
	Define class for automatic differentiation,
	overloading the basic operators
	"""
	def __init__(self, value: float, derivative: float):
		self.value = value
		self.derivative = derivative

	def __add__(self, other):
		if isinstance(other, Surreal):
			value = self.value + other.value
			derivative = self.derivative + other.derivative
		else:
			value = self.value + other
			derivative = self.derivative
		return Surreal(value, derivative)
	
	def __radd__(self, other):
		return self + other

	def __sub__(self, other):
		if isinstance(other, Surreal):
			value = self.value - other.value
			derivative = self.derivative - other.derivative
		else:
			value = self.value - other
			derivative = self.derivative
		return Surreal(value, derivative)
	
	def __rsub__(self, other):
		return -self + other

	def __mul__(self, other):
		if isinstance(other, Surreal):
			value = self.value * other.value
			derivative = self.derivative * other.value + self.value * other.derivative
		else:
			value = self.value * other
			derivative = self.derivative * other
		return Surreal(value, derivative)
	
	def __rmul__(self,other):
		return self * other
	
	def __neg__(self):
		return self*(-1)

	def __truediv__(self, other):
		if isinstance(other, Surreal):
			value = self.value / other.value
			derivative = self.derivative / other.value + self.value * -other.value**(-2) * other.derivative
		else:
			value = self.value / other
			derivative = self.derivative / other
		return Surreal(value, derivative)
	
	def __rtruediv__(self, other):
		return self / other
	
	def __pow__(self, other: int):
		if other == 0:
			return Surreal(1, 0)
		elif other == 1:
			return self
		elif other > 1:
			result = self
			for _ in range(other - 1):
				result = result * self
			return result

	def __eq__(self, other):
		if isinstance(other, Surreal):
			return self.value == other.value
		else:
			return self.value == other
	
	def __le__(self, other):
		if isinstance(other, Surreal):
			return self.value <= other.value
		else:
			return self.value <= other
	
	def __ge__(self, other):
		if isinstance(other, Surreal):
			return self.value >= other.value
		else:
			return self.value >= other
		
	def __le__(self, other):
		if isinstance(other, Surreal):
			return self.value <= other.value
		else:
			return self.value <= other
	
	def __lt__(self, other):
		if isinstance(other, Surreal):
			return self.value < other.value
		else:
			return self.value < other
		
	def __gt__(self, other):
		if isinstance(other, Surreal):
			return self.value > other.value
		else:
			return self.value > other
		
	def __str__(self):
		return f"[{self.value}, {self.derivative}]"
