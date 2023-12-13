import tkinter


class UserInterface:
    def __init__(self):
        self.app = tkinter.Tk()
        self.app.title("BTLab Camera Controller")
        self.app.protocol("WM_DELETE_WINDOW", self.quit)
        self.app.bind("<Control-q>", lambda _: self.quit())
        self.app.geometry("1280x720")

    def start(self):
        self.app.mainloop()

    def quit(self):
        self.app.quit()


if __name__ == "__main__":
    main = UserInterface()
    main.start()
