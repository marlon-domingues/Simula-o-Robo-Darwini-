#Arquivo de simulações do projeto Darwini 

import clases_Darwini as cla
import numpy as np 
import matplotlib.pyplot as plt

Darwini= cla.Robo("Darwini")
Fachada= cla.Fachada("Frontal",10, 30)

resolucaoV=500
resolucaoH=round(resolucaoV*Fachada.largura/Fachada.altura)

posH=np.linspace(1,Fachada.largura-1,resolucaoH)
posV=np.linspace(1,Fachada.altura-1,resolucaoV)

matrix_fachada = np.zeros((resolucaoV,resolucaoH))
camadas = np.array([matrix_fachada,matrix_fachada,matrix_fachada,matrix_fachada])

def main(Darwini, Fachada, camadas, posH, posV):
    cabos= cla.calc_surf_cabo(Darwini, Fachada, camadas, posH, posV, True)
    cla.derivada_zigzag(Fachada,camadas, posH, posV,resolucaoH,resolucaoV, Fachada.altura, True,True, True)
    angulos= cla.calc_surf_ang(Darwini, Fachada,camadas, posH, posV, True)
    F3,F4=cla.calc_surf_tension(Darwini,angulos, posH, posV, True)

main(Darwini, Fachada, camadas, posH, posV) 
