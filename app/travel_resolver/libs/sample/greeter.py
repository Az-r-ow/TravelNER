class GreeterHelper():
    def __init__(self):
        """
            Initialize base private attributes with sample values.
        """
        self._base_message = "Hello ! "

    def Greet(self, message: str):
        """
            Print greeting message in the terminal.

            Args:
                message (str): Message that will be included in greeting.

            Returns:
                (str): The full printed message.
        """
        print(self._base_message + message)
        return "Hello ! " + message
