class GreeterHelper():
    def __init__(self):
        self._base_message = "Hello ! "

    def Greet(self, message):
        print(self._base_message + message)
        return "Hello ! " + message