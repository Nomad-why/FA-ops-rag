class A():
    def a1(self):
        print('你好A')
    def b1(self):
        self.a1()

class B(A):
    def a1(self,):
        print('你好B')

b = B()
b.b1()

