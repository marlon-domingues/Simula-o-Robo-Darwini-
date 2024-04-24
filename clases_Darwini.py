#Arquivo de clases utilizadas no projeto Darwini 

import math
import matplotlib.pyplot as plt
import numpy as np 

class Robo():
    def __init__(self,nome):
        self.nome=nome 
        self.peso=20
        self.contrapeso=30
        self.cg=[0,0]
        self.mom_inercia=0
        self.pontos_fix=[[-0.5,0.5],[0.5,0.5],[0.5,-0.5],[-0.5,-0.5]]
        self.pontos_fix_d=[[-0.5,0.5],[0.5,0.5],[0.5,-0.5],[-0.5,-0.5]]
        self.pos=[0,0]
        self.beta=math.pi/4

class Fachada():
    def __init__(self,nome,largura, altura):
        self.nome=nome
        self.largura=largura
        self.altura =altura
        self.pontos_fix=[[0,self.altura],[self.largura,self.altura],[self.largura,0],[0,0]]
        self.area=largura*altura 

def sum_vetor(vet1,vet2):
    if len(vet1) == len(vet2):
        soma=[]
        for i in range(len(vet1)):
            soma.append(vet1[i]+vet2[i])

    return soma

def move( mov, robo, flag):  #Se a flag for True move o robo para a posição mov, caso contrário, desloca o robo a partir do ponto atual
    if flag:
        robo.pos=mov
    else:   
        robo.pos=sum_vetor(robo.pos,mov)

    for i in range(len(robo.pontos_fix_d)):
        robo.pontos_fix_d[i]= sum_vetor(robo.pontos_fix[i],robo.pos)

def calc_dist(pos1,pos2):
    if len(pos1)==2 and len(pos2)==2:
        difx=pos1[0]-pos2[0]
        dify=pos1[1]-pos2[1]
        dist=math.sqrt(difx**2+dify**2)

        return dist 
    
    else:
        print("vetor de tamanho inapropriado.")

def graf_3d(x,y,z, x_label, y_label,z_label,title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z)
    surf = ax.plot_surface(x, y, z, cmap='viridis')
    fig.colorbar(surf)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    plt.show()

def calc_surf_cabo(Robo, Fachada, cabos, posH, posV, flag):
    for i in range(len(cabos)):
        for j in range(len(cabos[i])):
            for k in range(len(cabos[i][j])):
                move([posH[k],posV[j]], Robo, True)
                cabos[i][j][k]=calc_dist(Fachada.pontos_fix[i],Robo.pontos_fix_d[i])

    if flag:
        x,y = np.meshgrid(posH,posV)
        graf_3d(x,y,cabos[0]+cabos[2],"Largura (m)","Comprimento (m)", "cabo (m)", "Soma dos Cabos da Diagonal 1 ")
        graf_3d(x,y,cabos[1]+cabos[3],"Largura (m)","Comprimento (m)", "cabo (m)", "Soma dos Cabos da Diagonal 2 ")
    
    return cabos

def derivada_zigzag(Fachada, cabos, posH, posV,resolucaoH,resolucaoV, zigzags, print_cabos,print_diag, print_dt):

    cabo_diagonal1=[]
    cabo_diagonal2=[]
    cabo3=[]
    cabo4=[]
    linha=0
    coluna=0
    direcao=1
    tempo=[0]

    while linha<len(posV)-round(len(posV)/zigzags):

        for i in range(len(posH)):

            if direcao==1:
                coluna=i
            else:
                coluna=len(posH)-i-1
            cabo3.append(cabos[2][linha][coluna])
            cabo4.append(cabos[3][linha][coluna])
            cabo_diagonal1.append(cabos[0][linha][coluna]+cabos[2][linha][coluna])
            cabo_diagonal2.append(cabos[1][linha][coluna]+cabos[3][linha][coluna])
            tempo.append(tempo[-1]+Fachada.largura/resolucaoH)

        direcao=direcao*-1
        
        for i in range(round(len(posV)/zigzags)):
            cabo3.append(cabos[2][linha][coluna])
            cabo4.append(cabos[3][linha][coluna])
            cabo_diagonal1.append(cabos[0][linha+i][coluna]+cabos[2][linha+i][coluna])
            cabo_diagonal2.append(cabos[1][linha+i][coluna]+cabos[3][linha+i][coluna])
            tempo.append(tempo[-1]+Fachada.altura/resolucaoV)

        linha=linha+i

    dt2_cabo_diagonal1=np.gradient(np.gradient(cabo_diagonal1))/(Fachada.largura/resolucaoH)**2
    dt2_cabo_diagonal2=np.gradient(np.gradient(cabo_diagonal2))/(Fachada.largura/resolucaoH)**2
    tempo=tempo[1:]

    if print_cabos:
        plt.plot(tempo,cabo3,label='Segmento de cabo 3', color='blue')
        plt.plot(tempo,cabo4,label='Segmento de cabo 4', color='red')
        plt.title('Comprimento dos Cabos em Uma Trajetória de Zig-zag')  
        plt.xlabel('Tempo (s)')               
        plt.ylabel('Comprimento (m)')                   
        plt.legend()
        plt.grid(True)
        plt.show()
        
    if print_diag:
        plt.plot(tempo,cabo_diagonal1,label='Soma da Diagonal 1', color='blue')
        plt.plot(tempo,cabo_diagonal2,label='Soma da Diagonal 2', color='red')
        plt.title('Soma das Diagonais em Uma Trajetória de Zig-zag')  
        plt.xlabel('Tempo (s)')               
        plt.ylabel('Comprimento (m)')                   
        plt.legend()
        plt.grid(True)
        plt.show()


    if print_dt:
        plt.plot(tempo,dt2_cabo_diagonal1, label='Aceleração do Contrapeso 1', color='blue')
        plt.plot(tempo,dt2_cabo_diagonal2, label='Aceleração do Contrapeso 2', color='red')
        plt.title('Aceleração dos Contrapesos em Uma Trajetória de Zig-zag')  
        plt.xlabel('Tempo (s)')                    
        plt.ylabel('Aceleração (m/s^2)')                    
        plt.legend()
        plt.grid(True)
        plt.show()

def calc_ang(pos1,pos2):
    if len(pos1)==2 and len(pos2)==2:
        difx=pos1[0]-pos2[0]
        dify=pos1[1]-pos2[1]
        ang=math.atan(abs(dify/difx))

        return ang
    
    else:
        print("vetor de tamanho inapropriado.")

def calc_surf_ang(Robo, Fachada, angulos, posH, posV, flag):
    for i in range(len(angulos)):
        for j in range(len(angulos[i])):
            for k in range(len(angulos[i][j])):
                move([posH[k],posV[j]], Robo, True)
                angulos[i][j][k]=calc_ang(Fachada.pontos_fix[i],Robo.pontos_fix_d[i])

    if flag:
        x,y = np.meshgrid(posH,posV)
        graf_3d(x,y,angulos[0],"Largura (m)","Comprimento (m)", "Angulo1 (rad)","Angulo 1 em Função da Posição")
        graf_3d(x,y,angulos[1],"Largura (m)","Comprimento (m)", "Angulo2 (rad)","Angulo 2 em Função da Posição")
        graf_3d(x,y,angulos[2],"Largura (m)","Comprimento (m)", "Angulo3 (rad)","Angulo 3 em Função da Posição")
        graf_3d(x,y,angulos[3],"Largura (m)","Comprimento (m)", "Angulo4 (rad)","Angulo 4 em Função da Posição")

    return angulos 

def calc_tension_inf(pos,Robo,angulos):

    ang1=angulos[0][pos[0]][pos[1]]
    ang2=angulos[1][pos[0]][pos[1]]
    ang3=angulos[2][pos[0]][pos[1]]
    ang4=angulos[3][pos[0]][pos[1]]
    beta=Robo.beta
    m=Robo.peso
    M=Robo.contrapeso
    g=9.81

    A=np.array([[-math.sin(ang3),-math.sin(ang4)],
                [2*math.cos(beta)-math.cos(ang3),-2*math.cos(beta)+math.cos(ang4)]])
    
    B=np.array([g*(m-M*(math.sin(ang1)+math.sin(ang2))),
                -M*g*(math.cos(ang1)-math.cos(ang2))])
    
    F= np.dot(np.linalg.inv(A), B)

    return F

def calc_surf_tension(Robo, angulos,posH,posV,flag):
    F3 = np.zeros((len(angulos[0]), len(angulos[0][0])))
    F4 = np.zeros((len(angulos[0]), len(angulos[0][0])))
    
    for j in range(len(angulos[0])):
        for k in range(len(angulos[0][j])):
            F3[j][k],F4[j][k]=calc_tension_inf((j,k),Robo,angulos)

    if flag:
        x,y = np.meshgrid(posH,posV)
        graf_3d(x,y,F3,"Largura (m)","Comprimento (m)", "Tensao (N)","Tensão Mecanica Estática no Segmento de Cabo 3")
        graf_3d(x,y,F4,"Largura (m)","Comprimento (m)", "Tensao (N)","Tensão Mecanica Estática no Segmento de Cabo 4")

    return F3,F4