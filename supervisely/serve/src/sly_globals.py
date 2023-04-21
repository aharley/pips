import os
import sys
import pathlib
import supervisely as sly
from dotenv import load_dotenv  # pip install python-dotenv\

logger = sly.logger

root_source_path = str(pathlib.Path(os.path.abspath(sys.argv[0])).parents[1])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

pips_root_path = str(pathlib.Path(os.path.abspath(sys.argv[0])).parents[3])
sys.path.append(pips_root_path)

sly.logger.info(f"Module source directory: {pips_root_path}")

# load_dotenv(os.path.join(pips_root_path, "supervisely/serve/debug.env"))
# load_dotenv(os.path.join(pips_root_path, "supervisely/serve/secret_debug.env"), override=True)

api = sly.Api()
my_app = sly.AppService()
# api = my_app.public_api
task_id = my_app.task_id

sly.fs.clean_dir(my_app.data_dir)  # @TODO: for debug

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
# device = os.environ['modal.state.device']
init_dir = os.environ.get("WEIGHTS_DIR", None)
if init_dir is None:
    init_dir = str(pathlib.Path(pips_root_path) / "reference_model")

local_info_dir = os.path.join(my_app.data_dir, "info")
sly.fs.mkdir(local_info_dir)


def get_files_paths(src_dir, extensions):
    files_paths = []
    for root, dirs, files in os.walk(src_dir):
        for extension in extensions:
            for file in files:
                if file.endswith(extension):
                    file_path = os.path.join(root, file)
                    files_paths.append(file_path)

    return files_paths