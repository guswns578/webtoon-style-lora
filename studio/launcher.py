import atexit,os,runpy,socket,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ['GRADIO_ANALYTICS_ENABLED']='False';os.environ['PYTHONDONTWRITEBYTECODE']='1'
for port in range(7871,7882):
 s=socket.socket()
 try:s.bind(('127.0.0.1',port));s.close();break
 except OSError:s.close()
else:raise SystemExit('7871~7881 포트가 사용 중입니다. 기존 실행 창을 확인하세요.')
os.environ['STUDIO_UI_PORT']=str(port)
try:
 import gradio,cv2,onnxruntime,numpy,PIL
 import forge_control
 atexit.register(forge_control.stop_owned)
 print('\n웹툰 캐릭터 스튜디오 시작\n주소: http://127.0.0.1:'+str(port)+'\n종료하려면 이 창에서 Ctrl+C를 누르세요.\n')
 runpy.run_path(str(ROOT/'app.py'),run_name='__main__')
except ModuleNotFoundError as e:
 print('필요한 패키지가 없습니다:',e,'\n먼저 install.bat를 실행하세요.');raise SystemExit(1)
except KeyboardInterrupt:print('종료합니다. 대기 작업은 다음 실행 때 이어집니다.')
