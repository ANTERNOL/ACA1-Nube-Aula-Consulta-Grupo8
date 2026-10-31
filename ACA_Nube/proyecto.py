"""Aula Consulta. Recuperación documental educativa, sin generación con LLM."""
from pathlib import Path
import os, json, csv, sqlite3, re, uuid, time, hashlib
from urllib.request import Request, urlopen
from urllib.error import HTTPError
BASE = Path(__file__).resolve().parent
MODELO = 'intfloat/multilingual-e5-small'
COLECCION = 'aula_consulta_e5_v1'

def base_sql():
    """Carga el dataset original y consulta sus agregados sin vincular personas a CUN."""
    import pandas as pd
    datos = pd.read_csv(BASE/'data/student-mat.csv', sep=';')
    assert len(datos)==395 and len(datos.columns)==33
    con=sqlite3.connect(BASE/'resultados/estudiantes.sqlite')
    datos.to_sql('matematicas',con,if_exists='replace',index=False)
    consultas={
      'resumen':'SELECT COUNT(*) estudiantes, ROUND(AVG(G3),2) promedio_G3 FROM matematicas',
      'por_apoyo_familiar':'SELECT famsup, COUNT(*) estudiantes, ROUND(AVG(G3),2) promedio_G3 FROM matematicas GROUP BY famsup',
      'por_tiempo_estudio':'SELECT studytime, COUNT(*) estudiantes, ROUND(AVG(G3),2) promedio_G3 FROM matematicas GROUP BY studytime ORDER BY studytime'}
    salida={nombre:pd.read_sql_query(sql,con).to_dict('records') for nombre,sql in consultas.items()}
    con.close()
    (BASE/'resultados/resumen_sql.json').write_text(json.dumps(salida,ensure_ascii=False,indent=2),encoding='utf-8')
    return salida

def cargar_paginas():
    from pypdf import PdfReader
    fuentes=json.loads((BASE/'fuentes.json').read_text(encoding='utf-8-sig'))
    paginas=[]
    for fuente in fuentes:
        archivo=BASE/'documentos'/fuente['archivo']
        sha=hashlib.sha256(archivo.read_bytes()).hexdigest()
        for n,p in enumerate(PdfReader(archivo).pages,1):
            texto=re.sub(r'\s+',' ',p.extract_text() or '').strip()
            if texto:
                paginas.append({'archivo':fuente['archivo'],'url':fuente['url'],'pagina':n,'texto':texto,'sha256':sha})
    return paginas

def temas(texto):
    t=texto.lower()
    return [k for k,v in {'schoolsup':['académic','tutor','acompañamiento'],'famsup':['familia'],'studytime':['estudio','aprendizaje'],'failures':['reprob','rendimiento','permanencia']}.items() if any(s in t for s in v)]

def fragmentar(paginas, tokenizador, tamano=220, solapamiento=40):
    """Ventanas de palabras, reducidas si exceden 460 tokens de E5. Respeta páginas."""
    if not 0 <= solapamiento < tamano: raise ValueError('Solapamiento inválido')
    salida=[]
    for pagina in paginas:
        palabras=pagina['texto'].split();inicio=0;indice=0
        while inicio<len(palabras):
            fin=min(inicio+tamano,len(palabras))
            while fin>inicio+1 and len(tokenizador.encode('passage: '+' '.join(palabras[inicio:fin]),add_special_tokens=True))>460:
                fin-=1
            texto=' '.join(palabras[inicio:fin])
            if not texto:break
            tokens=len(tokenizador.encode('passage: '+texto,add_special_tokens=True))
            if tokens>460:raise ValueError('Palabra demasiado larga para el límite del modelo')
            meta={k:v for k,v in pagina.items() if k!='texto'}
            meta.update(texto=texto,fragmento=indice,tokens=tokens,temas_dataset=temas(texto),dataset='UCI Student Performance / Matemáticas',relacion='temática; no existe vínculo individual ni equivalencia normativa',modelo=MODELO)
            meta['id']=str(uuid.uuid5(uuid.NAMESPACE_URL,pagina['sha256']+':'+str(pagina['pagina'])+':'+str(inicio)+':'+str(fin)))
            salida.append(meta)
            if fin==len(palabras):break
            inicio=max(inicio+1,fin-solapamiento);indice+=1
    return salida

def peticion(metodo,ruta,cuerpo=None):
    """Conexión REST HTTPS remota. Nunca imprime la clave ni respuestas de error."""
    url=os.getenv('QDRANT_URL','').rstrip('/');clave=os.getenv('QDRANT_API_KEY','')
    if not url.startswith('https://') or not clave:raise RuntimeError('Configure QDRANT_URL HTTPS y QDRANT_API_KEY')
    raw=None if cuerpo is None else json.dumps(cuerpo).encode()
    req=Request(url+ruta,data=raw,method=metodo,headers={'api-key':clave,'Content-Type':'application/json'})
    try:
        with urlopen(req,timeout=90) as r:return json.load(r)
    except HTTPError as e:raise RuntimeError(f'Qdrant respondió HTTP {e.code}. Revise permisos y configuración.') from None

def conectar():
    return peticion('GET','/collections')

def modelo_local():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODELO,device='cpu')

def indexar(modelo,fragmentos):
    existentes=conectar()['result']['collections']
    if not any(x['name']==COLECCION for x in existentes):
        peticion('PUT','/collections/'+COLECCION,{'vectors':{'size':384,'distance':'Cosine'}})
    config=peticion('GET','/collections/'+COLECCION)['result']['config']['params']['vectors']
    if config.get('size')!=384 or config.get('distance')!='Cosine':raise RuntimeError('La colección existente no corresponde al modelo E5')
    vectores=modelo.encode(['passage: '+x['texto'] for x in fragmentos],normalize_embeddings=True,batch_size=16,show_progress_bar=True)
    for inicio in range(0,len(fragmentos),32):
        puntos=[{'id':x['id'],'vector':v.tolist(),'payload':x} for x,v in zip(fragmentos[inicio:inicio+32],vectores[inicio:inicio+32])]
        peticion('PUT','/collections/'+COLECCION+'/points?wait=true',{'points':puntos})
    conteo=peticion('POST','/collections/'+COLECCION+'/points/count',{'exact':True})['result']['count']
    reporte={'coleccion':COLECCION,'fragmentos_corpus':len(fragmentos),'puntos_remotos':conteo,'dimension':384,'distancia':'Cosine','modelo':MODELO,'tamano_palabras':220,'solapamiento_palabras':40,'max_tokens':max(x['tokens'] for x in fragmentos),'relacion_dataset':'temática, no individual'}
    (BASE/'resultados/indexacion.json').write_text(json.dumps(reporte,indent=2,ensure_ascii=False))
    (BASE/'resultados/fragmentos.json').write_text(json.dumps(fragmentos,indent=2,ensure_ascii=False))
    return reporte

def consultar(modelo,pregunta,k=3):
    if not pregunta.strip():raise ValueError('Escriba una pregunta')
    vector=modelo.encode(['query: '+pregunta],normalize_embeddings=True)[0].tolist()
    inicio=time.perf_counter()
    r=peticion('POST','/collections/'+COLECCION+'/points/query',{'query':vector,'limit':k,'with_payload':True,'with_vector':True})
    segundos=time.perf_counter()-inicio
    import numpy as np
    filas=[]
    for rango,p in enumerate(r['result']['points'],1):
        v=np.asarray(p['vector']);q=np.asarray(vector)
        coseno=float(np.dot(q,v)/(np.linalg.norm(q)*np.linalg.norm(v)))
        m=p['payload']
        filas.append({'pregunta':pregunta,'rango':rango,'similitud':p['score'],'coseno_recalculado':coseno,'archivo':m['archivo'],'pagina':m['pagina'],'texto':m['texto'],'url':m['url'],'latencia_remota_s':segundos})
    return filas

PREGUNTAS=[
 '¿Qué derechos tienen los estudiantes en la institución?',
 '¿Qué acompañamiento académico ofrece bienestar a los estudiantes?',
 '¿Cómo participa la familia en el bienestar de la comunidad educativa?',
 '¿Qué apoyos se contemplan frente a dificultades económicas?',
 '¿Cuál es la receta de una pizza margarita?']

def validar(modelo):
    filas=[f for p in PREGUNTAS for f in consultar(modelo,p)]
    import pandas as pd
    df=pd.DataFrame(filas);df.to_csv(BASE/'resultados/consultas.csv',index=False)
    (BASE/'resultados/consultas.json').write_text(json.dumps(filas,ensure_ascii=False,indent=2),encoding='utf-8')
    return df

def preparar_carpetas():
    (BASE/'resultados').mkdir(exist_ok=True)

if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv(BASE/'.env',encoding='utf-8-sig',override=True);preparar_carpetas()
    print('Resumen SQL:',base_sql())
    conectar();print('Conexión remota confirmada')
    m=modelo_local();fragmentos=fragmentar(cargar_paginas(),m.tokenizer)
    print(indexar(m,fragmentos));print(validar(m)[['pregunta','rango','similitud','archivo','pagina']].to_string(index=False))
