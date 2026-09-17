"""Community Cloud entry point for the frozen review app."""
import os
import runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['NFL_MODEL_DATA_DIR']=str(ROOT/'data')
os.environ['NFL_ARTIFACT_MAP']=str(ROOT/'artifact-map.json')
os.environ['NFL_REVIEW_BUILD']='1'
runpy.run_path(str(ROOT/'app'/'streamlit_app.py'),run_name='__main__')
