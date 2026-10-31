"""Consulta remota sin reindexar. No muestra credenciales."""
from dotenv import load_dotenv
from proyecto import BASE, conectar, modelo_local, consultar
load_dotenv(BASE/'.env',encoding='utf-8-sig',override=True)
print('AULA CONSULTA - GRUPO 8')
conectar()
print('Conexion remota confirmada. Cargando modelo...')
m=modelo_local()
while True:
    p=input('\nEscriba su pregunta (Enter para salir): ').strip()
    if not p:break
    try:
        filas=consultar(m,p)
        for f in filas:
            print('\nRESULTADO',f['rango'],'- Similitud:',round(f['similitud'],4))
            print('Documento:',f['archivo'],'- Pagina del PDF:',f['pagina'])
            print(f['texto'])
            print('Fuente:',f['url'])
        if filas:print('\nTiempo de consulta remota:',round(filas[0]['latencia_remota_s'],3),'s')
        print('Revise si el texto responde. La similitud no es probabilidad de acierto.')
    except Exception as e:
        print('Consulta interrumpida:',type(e).__name__)
        print('Revise conexion y configuracion antes de repetir.')
