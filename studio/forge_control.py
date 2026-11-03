"""Start only the Forge instance owned by this app; never stop unrelated processes."""
import json,os,subprocess,threading,time
from pathlib import Path
import engine
PROCESS=None
def start():
 global PROCESS
 try:engine.api('/sdapi/v1/options',timeout=3);return '이미 실행 중인 Forge에 연결했습니다.'
 except Exception:pass
 if PROCESS is not None and PROCESS.poll() is None:return 'Forge를 이미 시작하는 중입니다. 잠시 후 다시 점검을 누르세요.'
 cfg=engine.settings();forge=Path(cfg['forge_dir']);models=Path(cfg['models_dir'])
 launch=forge/'launch.py';python=forge/'venv'/'Scripts'/'python.exe'
 if not python.is_file():python=forge/'.venv'/'Scripts'/'python.exe'
 if not launch.is_file() or not python.is_file():raise ValueError('Forge 설치 폴더를 지정하세요. launch.py와 venv의 Python이 있어야 합니다.')
 if not models.is_dir():raise ValueError('모델 폴더를 찾지 못했습니다. 전달된 models 폴더를 지정하세요.')
 port=int(cfg['api_url'].rsplit(':',1)[1]);runtime=engine.RUNTIME/'forge';runtime.mkdir(exist_ok=True)
 settings=runtime/'config.json'
 # Let Forge initialize its own versioned configuration on the first run.
 env=os.environ.copy();env.update(SD_WEBUI_CACHE_DIR=str(runtime/'cache'),PYTHONDONTWRITEBYTECODE='1',GRADIO_ANALYTICS_ENABLED='False')
 args=[str(python),'-u',str(launch),'--api','--nowebui','--server-name','127.0.0.1','--port',str(port),'--skip-python-version-check','--skip-prepare-environment','--skip-install','--disable-extra-extensions','--data-dir',str(runtime),'--model-ref',str(models),'--ui-settings-file',str(settings)]
 with (runtime/'forge.log').open('a',encoding='utf-8') as log:
  PROCESS=subprocess.Popen(args,cwd=forge,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 return 'Forge를 시작했습니다. 처음에는 모델을 불러오는 데 시간이 걸립니다. 잠시 후 환경 점검을 다시 누르세요.'
def status():
 if PROCESS is None:return '이 도구에서 시작한 Forge 프로세스가 없습니다.'
 return '이 도구에서 시작한 Forge가 실행 중입니다.' if PROCESS.poll() is None else 'Forge가 종료됐습니다. runtime/forge/forge.log를 확인하세요.'
def stop_owned():
 if PROCESS is not None and PROCESS.poll() is None:
  PROCESS.terminate()
  try:PROCESS.wait(timeout=10)
  except subprocess.TimeoutExpired:PROCESS.kill()
