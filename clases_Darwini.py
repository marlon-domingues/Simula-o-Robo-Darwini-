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
        self.d=0.5
        self.pontos_fix=[[-self.d,self.d],[self.d,self.d],[self.d,-self.d],[-self.d,-self.d]]
        self.pontos_fix_d=[[-self.d,self.d],[self.d,self.d],[self.d,-self.d],[-self.d,-self.d]]
        self.pos=[0,0]
        self.beta=math.pi/4

class Fachada():
    def __init__(self,nome,largura, altura):
        self.nome=nome
        self.largura=largura
        self.altura =altura
        self.pontos_fix=[[0,self.altura],[self.largura,self.altura],[self.largura,0],[0,0]]
        self.area=largura*altura 

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = 0
        self.prev_error = 0
        self.integral = 0

    def update(self, measured_value,setpoint):
        self.setpoint=setpoint
        error = self.setpoint - measured_value

        # P term
        proportional = self.kp * error

        # I term
        self.integral += error
        integral = self.ki * self.integral

        # D term
        derivative = self.kd * (error - self.prev_error)
        self.prev_error = error

        output = proportional + integral + derivative

        return output, error
    
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

def calc_surf_cabo(Robo, Fachada, cabos, posH, posV, flag): #Calcula o comprimento da cada cabo no plano 
    for i in range(len(cabos)):
        for j in range(len(cabos[i])):
            for k in range(len(cabos[i][j])):
                move([posH[k],posV[j]], Robo, True)
                cabos[i][j][k]=calc_dist(Fachada.pontos_fix[i],Robo.pontos_fix_d[i])

    if flag:
        x,y = np.meshgrid(posH,posV)
        graf_3d(x,y,cabos[2]+cabos[0],"Largura (m)","Comprimento (m)", "cabo (m)", "Soma dos Cabos da Diagonal 1 ")
        graf_3d(x,y,cabos[3]+cabos[1],"Largura (m)","Comprimento (m)", "cabo (m)", "Soma dos Cabos da Diagonal 2 ")
    
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

def controle_move(trajetoria,pos_init,Fachada, Robo, flag_trajetoria):
    lfuturo=[[0],[0]]
    latual=[[0],[0]]
    kp=0.4
    ki=0.1
    kd=0.3
    x=[]
    y=[]

    PID_motorD=PIDController(kp,ki,kd)
    PID_motorE=PIDController(kp,ki,kd)

    move([pos_init[0],pos_init[1]], Robo, True)

    for i in range(2):
        latual[i]=calc_dist(Fachada.pontos_fix[i+2],Robo.pontos_fix_d[i+2])

    for i in range(len(trajetoria[0])):

        move([trajetoria[0][i],trajetoria[1][i]], Robo, True)

        for i in range(2):
            lfuturo[i]=calc_dist(Fachada.pontos_fix[i+2],Robo.pontos_fix_d[i+2])

        errorD =1
        errorE =1
        caboe=[]
        cabod=[]

        control_signalE, errorE= PID_motorE.update(latual[0],lfuturo[0])
        control_signalD, errorD = PID_motorD.update(latual[1],lfuturo[1])

        latual[0]+=control_signalE 
        latual[1]+=control_signalD 

        base=Fachada.largura-Robo.d*2
        a=(base**2+latual[1]**2-latual[0]**2)/(2*base)
        b=np.sqrt(latual[1]**2-a**2)
        x.append(a+Robo.d)
        y.append(b+Robo.d)
        caboe.append(latual[1])
        cabod.append(latual[0])
    

    if flag_trajetoria:
        plt.plot(trajetoria[0], trajetoria[1], color='red', label='Referência', linewidth=4)
        plt.plot(x, y, color='blue', label='Robo')
        plt.xlabel('Eixo X')
        plt.ylabel('Eixo Y')
        plt.title('Trajetória do Robô')
        plt.legend()
        plt.grid(True)
        plt.show()

    dist=[]
    for i in range(len(x)-1):
        dist.append(np.sqrt((x[i+1]-x[i])**2+(y[i+1]-y[i])**2)*100)
    
    plt.plot(dist)
    plt.show()

    
    return x,y,caboe,cabod
    

def plot_trajetoria(x,y):
    
    plt.plot(x, y, color='blue', label='Pontos')
    plt.xlabel('Eixo X')
    plt.ylabel('Eixo Y')
    plt.title('Trajetória do Robô')
    plt.legend()
    plt.grid(True)
    plt.show()

def controle_move_step(pos_ref,pos_init,Fachada, Robo, flag_trajetoria,discretizacao,erro):
    vetordir=np.array(pos_ref)-np.array(pos_init)
    mod_vetordir=np.sqrt(vetordir[0]**2+vetordir[1]**2)
    ciclos=round(mod_vetordir/discretizacao)
    vetor_mov=vetordir/(ciclos)
    vetorref=pos_init+vetor_mov
    x=[]
    y=[]
    caboe=[]
    cabod=[]

    for _ in range(ciclos):
        xp,yp,ce,cd=controle_move(vetorref,pos_init,Fachada,Robo,False,erro)
        vetorref+=vetor_mov
        pos_init+=vetor_mov
        x+=xp
        y+=yp
        caboe+=ce
        cabod+=cd


    xp,yp,ce,cd=controle_move(pos_ref,pos_init,Fachada,Robo,False,erro)
    x+=xp
    y+=yp
    caboe+=ce
    cabod+=cd

    dist=[]
    for i in range(len(x)-1):
        dist.append(np.sqrt((x[i+1]-x[i])**2+(y[i+1]-y[i])**2))
    
    if flag_trajetoria:
        plot_trajetoria(x,y)
    
    plt.plot(dist)
    plt.show()

    plt.plot(cabod)
    plt.plot(caboe)
    plt.show()

def trajetoria_zigzag(Fachada,zigzags):

    linha=[1]
    coluna=[1]
    direcao=1
    vel=1
    dt=0.01
    R=(Fachada.altura-2)/(zigzags*2)
    l=(Fachada.altura-2)/(zigzags)

    while linha[-1]<Fachada.altura-1:

        for i in range(round((Fachada.largura-2)/(vel*dt))):

            if direcao==1:
                coluna.append(coluna[-1]+vel*dt)
            else:
                coluna.append(coluna[-1]-vel*dt)
            
            linha.append(linha[-1])
        """
        for i in range(round(l/(vel*dt))):

            linha.append(linha[-1]+vel*dt)
            coluna.append(coluna[-1])
        """  

        linha_fim=linha[-1]
        coluna_fim=coluna[-1]
        tot_circle=round(math.pi*R/(vel*dt))

        for i in range(tot_circle-1):
           i+=1
           linha.append(R-math.cos(math.pi*i/tot_circle)*R+linha_fim)
           coluna.append(direcao*(math.sin(math.pi*i/tot_circle)*R)+coluna_fim)

        direcao=direcao*-1

    return [coluna,linha]


