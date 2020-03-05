#!"C:\Users\Cade Norman\PycharmProjects\EcoSight\venv\Scripts\python.exe"
# EASY-INSTALL-ENTRY-SCRIPT: 'mahotas==1.4.9','console_scripts','mahotas-features'
__requires__ = 'mahotas==1.4.9'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('mahotas==1.4.9', 'console_scripts', 'mahotas-features')()
    )
