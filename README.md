### Packaging

Linux (assuming tk and tcl are installed)

Install Python 3.11 then:
```bash
pip install pipenv

pipenv shell
pipenv install -d

pyinstaller --noconfirm --onedir --windowed \
  --hidden-import=tkinter --hidden-import=tkinter.filedialog --hidden-import=tkinter.font --hidden-import=tkinter.ttk \
  --hidden-import='PIL._tkinter_finder' \
  ./app/main.py
```
