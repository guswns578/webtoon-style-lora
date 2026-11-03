"""UI helpers. No GPU calls and no changes to model generation settings."""
import json
import numpy as np
from PIL import Image,ImageDraw,ImageOps
import engine as e

MASK_OPACITY=100

def editor_value(raw,mask,color=(32,156,232)):
 raw=raw.convert('RGBA')
 alpha=Image.fromarray(np.where(np.asarray(mask.convert('L'))>=128,MASK_OPACITY,0).astype('uint8'))
 layer=Image.new('RGBA',raw.size,(*color,0));layer.putalpha(alpha)
 return {'background':raw,'layers':[layer],'composite':Image.alpha_composite(raw,layer)}

def editor_mask(value,size):
 if not isinstance(value,dict):raise ValueError('마스크 편집 화면을 먼저 불러오세요.')
 masks=[]
 for layer in value.get('layers',[]):
  if isinstance(layer,str):
   with Image.open(layer) as image:layer=image.convert('RGBA')
  if layer is not None:masks.append(np.asarray(layer.convert('RGBA').resize(size,Image.Resampling.NEAREST).getchannel('A')))
 if not masks:return Image.new('L',size,0)
 # Masks are binary; display transparency never weakens the saved omission mask.
 return Image.fromarray(np.where(np.maximum.reduce(masks)>=MASK_OPACITY//2,255,0).astype('uint8'))

def layout_preview(bheight,cheight,bgmode,bgimage,foreground):
 figure=Image.new('RGBA',(300,1000));d=ImageDraw.Draw(figure);fill=(75,102,142,255)
 d.ellipse((95,15,205,155),fill=fill)
 d.rounded_rectangle((78,155,222,515),radius=35,fill=fill)
 d.line((92,205,30,465),fill=fill,width=45);d.line((208,205,270,465),fill=fill,width=45)
 d.line((112,495,85,955),fill=fill,width=65);d.line((188,495,215,955),fill=fill,width=65)
 params={'b_height':bheight,'c_height':cheight,'background_mode':bgmode,'foreground':foreground}
 if bgmode=='upload' and bgimage is not None:params['_preview_background']=bgimage
 b,_,bl=e.render(figure,'B',params);c,_,cl=e.render(figure,'C',params)
 caption=f'**B 인물 높이 {bheight:g}% · C 인물 높이 {cheight:g}%**'
 caption+='\n\n크기·배치 설명용 도형입니다. 선택한 캐릭터의 생성 예측이 아닙니다. B는 컷 아래가 잘릴 수 있고, 넓은 자세는 실제 합성 때 더 작아질 수 있습니다.'
 if bgmode=='upload' and bgimage is None:caption+=' 배경 이미지를 올리면 미리보기에도 반영됩니다.'
 return b,c,caption

def seed_value(mode,number):
 if mode=='verified':return -2
 if mode=='random':return -1
 if number is None or int(number)<0 or float(number)!=int(number):raise ValueError('직접 지정하는 시드는 0 이상의 정수로 입력하세요.')
 return int(number)

def progress_text(ids=None):
 rows=[e.job(i) for i in ids] if ids else e.jobs()
 rows=[j for j in rows if j]
 counts={s:sum(j['status']==s for j in rows) for s in ['queued','running','review','approved','rejected','failed','cancelled']}
 done=sum(counts[s] for s in ['review','approved','rejected'])
 prefix='이번 생성' if ids else '전체 작업'
 line=f"**{prefix} · 완료 {done} / {len(rows)} · 대기 {counts['queued']} · 생성 중 {counts['running']} · 실패 {counts['failed']} · 취소 {counts['cancelled']}**"
 active=next((j for j in rows if j['status']=='running'),None)
 if active:
  p=json.loads(active['params']);line+='\n\n'+e.PRESETS['characters'][p['character']]['name']+' → '+(active['stage'] or '준비')
 elif counts['queued']:line+='\n\n앞선 작업이 끝나면 순서대로 생성합니다.'
 elif ids:line+='\n\n생성이 끝났습니다. 결과 검수에서 확인하세요.'
 else:line+='\n\n캐릭터와 포즈를 선택해 생성하세요.'
 return line
