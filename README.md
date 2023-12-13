### Packaging

Linux (assuming tk and tcl are installed)

Install Python 3.11 and [Vimba SDK](https://www.alliedvision.com/en/products/vimba-sdk/), then:
```bash
pip install pipenv

pipenv shell
pipenv install -d

pyinstaller --noconfirm --onedir --windowed \
  --hidden-import=tkinter --hidden-import=tkinter.filedialog --hidden-import=tkinter.font --hidden-import=tkinter.ttk \
  --hidden-import='PIL._tkinter_finder' \
  --add-data "$(pip show customtkinter | grep -i location | awk '{ print $NF }')/customtkinter:customtkinter/" \
  --add-data "$(pip show darkdetect | grep -i location | awk '{ print $NF }')/darkdetect:darkdetect/" \
  ./app/main.py
```
