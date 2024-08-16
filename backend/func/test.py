# Parent class
class Parent:
    def __init__(self):
        self.name = "name"
        self.age = 11
    
    

class Child(Parent):
    def __init__(self):
        # Automatically inherit and initialize 'name' and 'age'
        super().__init__()
        pass
# Create an instance of Child
child = Child()

# Access inherited attributes
print(child.name)