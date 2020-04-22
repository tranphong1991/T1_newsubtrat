from numpy import *

#function calculate temperature profile Crank-Nicolson method
def solve_tc_numerique(k1, k2, rho1, rho2, c1, c2, L, R, f, sigma1, sigma2, h, Rj, I0, alpha, G, T_amb, N,J):
    #constant values
	S = pi*R**2
	p = 2*pi*R
	w = 2*pi*f
	rhoc_moyen = (rho1*c1 + rho2*c2)/2
	sigma = (sigma1 + sigma2)/2
	
	#setup grids
	dx = L/(N-1)
	x_grid = array([j * dx for j in range(N)])  #
    
	# setup t grid
	time = 20 / f
	dt = float(time) / float(J - 1)
	t_grid = array([n * dt for n in range(J)])

	# initial condition
	T = zeros((N, J))
	T[:, 0] = T_amb * ones(N)
	T2w_amp = zeros(N)

	# setup matrix b and solve linear system
	b = zeros(N)
	for j in range(0, J - 1):
		I = I0*cos(w*j*dt)
		
		# setup matrix A
		Aii1 = 1 + k1*dt/(rho1*c1*(dx** 2)) + h*p*dt/(rho1*c1*S*2) - sigma1*alpha*I**2*dt/(2*rho1*c1*S**2) # - sigma1_const*dt*alpha*I**2/(2*rho1_const*c1*S_const**2)
		Aii2 = 1 + k2*dt/(rho2*c2*(dx** 2)) + h*p*dt/(rho2*c2*S*2) - sigma2*alpha*I**2*dt/(2*rho2*c2*S**2)  # - sigma2_const*dt*alpha*I**2/(2*rho2_const*c2*S_const**2)
		Aii1_m1 = -k1 * dt / (2 * rho1 * c1 * (dx ** 2))
		Aii2_m1 = -k2 * dt / (2 * rho2 * c2 * (dx ** 2))
		Aii1_p1 = -k1 * dt / (2 * rho1 * c1 * (dx ** 2))
		Aii2_p1 = -k2 * dt / (2 * rho2 * c2 * (dx ** 2))
		AN2 = 1 + (k2 + k1)*dt/(2*(dx**2)*rhoc_moyen) + h*p*dt/(rhoc_moyen*S*2) + G*dt/(2*rhoc_moyen*S*dx) - sigma*alpha*I**2*dt/(2*rhoc_moyen*S**2)# - sigma2_const*dt*alpha*I**2/(2*rhoc_moyen*S_const**2)
		AN2_m1 = -k1 * dt / (2 * (dx ** 2) * rho1 * c1)
		AN2_p1 = -k2 * dt / (2 * (dx ** 2) * rho2 * c2)
		A = diagflat([0] + [Aii1_m1 for i in range(int((N - 1) / 2) - 2)] + [AN2_m1] + [Aii2_m1 for i in range(int((N - 1) / 2))],1) + \
			diagflat([1] + [Aii1 for i in range(int((N - 1) / 2) - 1)] + [AN2] + [Aii2 for i in range(int((N - 1) / 2) - 1)] + [1]) + \
			diagflat([Aii1_p1 for i in range(int((N - 1) / 2))] + [AN2_p1] + [Aii2_p1 for i in range(int((N - 1) / 2) - 2)] + [0], -1)

		# setup vector b
		b[0] = T[0, j]
		for i in range(1, N - 1):
			if i < (N - 1) / 2:
				b[i] = T[i, j] + k1 * dt * (T[i - 1, j] + T[i + 1, j] - 2 * T[i, j]) / (2 * (dx ** 2) * rho1 * c1) + \
				sigma1 * I ** 2 * dt / (rho1 * c1 * S ** 2) + dt * h * p * (2 * T_amb - T[i, j]) / (2 * rho1 * c1 * S) + sigma1*alpha*(T[i, j]-2*T_amb )*I**2*dt/(2*rho1*c1*S**2)
			elif i > (N - 1) / 2:
				b[i] = T[i, j] + k2 * dt * (T[i - 1, j] + T[i + 1, j] - 2 * T[i, j]) / (2 * (dx ** 2) * rho2 * c2) + \
				sigma2 * I ** 2 * dt / (rho2 * c2 * S ** 2) + dt * h * p * (2 * T_amb - T[i, j]) / (2 * rho2 * c2 * S) + sigma2*alpha*(T[i, j]-2*T_amb )*I**2*dt/(2*rho2*c2*S**2) 
			else:
                # b[i] = T[i,j]+(k2*dt*(T[i+1,j]-T[i,j]))/(2*(dx**2)*rho2*c2) + k1*dt*(T[i-1,j]-T[i,j])/(2*(dx**2)*rho1*c1) \
				b[i] = T[i, j] + (k2 * dt * (T[i + 1, j] - T[i, j]) + k1 * dt * (T[i - 1, j] - T[i, j])) / (2 * (dx ** 2) * rhoc_moyen)+ \
				sigma * I ** 2 * dt / (rhoc_moyen * S ** 2) + (Rj * I ** 2) * dt / (rhoc_moyen * S * dx)+ dt * h * p * (2 * T_amb - T[i, j]) / (2 * rhoc_moyen * S) +\
				G * (2 * T_amb - T[i, j]) * dt / (2 * rhoc_moyen * S * dx) + sigma*alpha*(T[i, j]-2*T_amb )*I**2*dt/(2*rhoc_moyen * S ** 2)
		b[N - 1] = T[N - 1, j]
		T[:, j + 1] = linalg.solve(A, b)
	
	#Fourrier transform to get T2w values searching T2w value	
	for i in range(0, N-1):
		T2w = T[i,:]
		fft_T = fft.fft(T2w)
		M = len(fft_T)
		f_s = 1.0/dt
		freq = fft.fftfreq(M,dt)
		half_m = int(ceil(M/2.0))
		fft_T_half = (2.0/M)*fft_T[:half_m]
		freq_half = freq[:half_m]
		T2w_amp[i] = abs(fft_T_half[3:half_m]).max()
	
	T2w0 = T2w_amp[int((N-1)/2)]
	T3w0 = trapz(T2w_amp,x_grid)/L
	return [T,T2w_amp,T2w0,T3w0,x_grid,t_grid]
	

#function simulation analytical model 2 wires
def solve_tc_anlytique_1f(k, rho, c, L, R, f, sigma, h, Rj, I0, G, T_amb,N):
	x_array = linspace(-L,L,N)
	T2w = zeros(N,'complex')
	w = 2*pi*f
	S = pi*R**2
	p = 2*pi*R
	
	m = sqrt((h*p)/(k*S)+2*1.j*(w*rho*c)/k)
	
	
	for i in range(0,N-1):
		T2w[i] = I0**2*(sigma*(-G + 2*S*k*m + (G + 2*S*k*m)*exp(2*L*m))*exp(m*abs(x_array[i])) + (sigma*(G - 2*S*k*m) + (-G*sigma + Rj*S**2*k*m**2)*exp(L*m))*exp(L*m) + \
		(G*sigma - Rj*S**2*k*m**2 - sigma*(G + 2*S*k*m)*exp(L*m))*exp(2*m*abs(x_array[i])))*exp(-m*abs(x_array[i]))/(2*S**2*k*m**2*(-G + 2*S*k*m + (G + 2*S*k*m)*exp(2*L*m)))
	
	T2w0 = T2w[int((N-1)/2)]
	T3w0 = trapz(T2w,x_array)/(2*L)
	return [abs(T2w),abs(T2w0),abs(T3w0),x_array]
	
	
#function simulation analytical model 1 wire simple
def tc_anal_ampl_phase(k, rho, c, L, R, f, sigma, h, Rj, I0, G, T_amb,N):
	x_array = linspace(-L,L,N)
	T2w = zeros(N,'complex')
	w = 2*pi*f
	S = pi*R**2
	p = 2*pi*R
	
	m = sqrt((h*p)/(k*S)+2*1.j*(w*rho*c)/k)
	
	
	for i in range(0,N-1):
		T2w[i] = I0**2*(sigma*(-G + 2*S*k*m + (G + 2*S*k*m)*exp(2*L*m))*exp(m*abs(x_array[i])) + (sigma*(G - 2*S*k*m) + (-G*sigma + Rj*S**2*k*m**2)*exp(L*m))*exp(L*m) + \
		(G*sigma - Rj*S**2*k*m**2 - sigma*(G + 2*S*k*m)*exp(L*m))*exp(2*m*abs(x_array[i])))*exp(-m*abs(x_array[i]))/(2*S**2*k*m**2*(-G + 2*S*k*m + (G + 2*S*k*m)*exp(2*L*m)))
	
	T2w0 = T2w[int((N-1)/2)]
	T3w0 = trapz(T2w,x_array)/(2*L)
	return [abs(T2w),abs(T2w0),abs(T3w0),x_array]

	
#function simulation analytical model 2 wires
def solve_tc_anlytique_2f(k1, k2, rho1, rho2, c1, c2, L, R, f, sigma1, sigma2, h, Rj, I0, G, T_amb,N):
	x_array = linspace(-L,L,N)
	T2w = zeros(N,'complex')
	w = 2*pi*f
	S = pi*R**2
	p = 2*pi*R
	
	m1 = sqrt((h*p)/(k1*S)+2*1.j*(w*rho1*c1)/k1)
	m2 = sqrt((h*p)/(k2*S)+2*1.j*(w*rho2*c2)/k2)
	
	for i in range(0,N-1):
		if x_array[i] < 0:
			T2w[i] = I0**2*(-2*S*k1*m1**2*sigma2*exp(L*m2) + S*(k1*m1**2*sigma2 - k2*m2**2*sigma1)*(exp(2*L*m2) + 1) + m2*sigma1*(-G - S*k1*m1 + S*k2*m2*exp(2*L*m2) + \
			S*k2*m2 + (G + S*k1*m1)*exp(2*L*m2))*exp(L*m1) + m2*sigma1*(-G + S*k1*m1 - S*k2*m2*exp(2*L*m1) + S*k2*m2*exp(2*L*m2) - S*k2*m2*exp(2*L*(m1 + m2)) + \
			S*k2*m2 + (G - S*k1*m1)*exp(2*L*m2) + (G + S*k1*m1)*exp(2*L*m1) - (G + S*k1*m1)*exp(2*L*(m1 + m2)))*exp(m1*x_array[i]) - m2*(G*sigma1 - Rj*S**2*k1*m1**2)*(exp(2*L*m2) - 1) - (-2*S*k1*m1**2*sigma2*exp(L*(2*m1 + m2)) + \
			S*(k1*m1**2*sigma2 - k2*m2**2*sigma1)*(exp(2*L*m1) + exp(2*L*(m1 + m2))) + m2*sigma1*(-G + S*k1*m1 + S*k2*m2*exp(2*L*m2) + S*k2*m2 + (G - S*k1*m1)*exp(2*L*m2))*exp(L*m1) + \
			m2*(G*sigma1 - Rj*S**2*k1*m1**2)*(exp(2*L*m1) - exp(2*L*(m1 + m2))))*exp(2*m1*x_array[i]))*exp(-m1*x_array[i])/(2*S**2*k1*m1**2*m2*(-G + S*k1*m1 - S*k2*m2*exp(2*L*m1) + S*k2*m2*exp(2*L*m2) - \
			S*k2*m2*exp(2*L*(m1 + m2)) + S*k2*m2 + (G - S*k1*m1)*exp(2*L*m2) + (G + S*k1*m1)*exp(2*L*m1) - (G + S*k1*m1)*exp(2*L*(m1 + m2))))
		elif x_array[i] >= 0:
			T2w[i] = I0**2*(2*S*k1*k2*m1*m2**2*sigma1*exp(L*(m1 + 2*m2)) - k1*m1**2*sigma2*(-G + S*k1*m1 - S*k2*m2*exp(2*L*m1) + S*k2*m2 + (G + S*k1*m1)*exp(2*L*m1))*exp(L*m2) + \
			k1*m1**2*sigma2*(-G + S*k1*m1 - S*k2*m2*exp(2*L*m1) + S*k2*m2*exp(2*L*m2) - S*k2*m2*exp(2*L*(m1 + m2)) + S*k2*m2 + (G - S*k1*m1)*exp(2*L*m2) + (G + S*k1*m1)*exp(2*L*m1) - \
			(G + S*k1*m1)*exp(2*L*(m1 + m2)))*exp(m2*x_array[i]) - k2*m2**2*(G*sigma1 - Rj*S**2*k1*m1**2)*(exp(2*L*m2) - exp(2*L*(m1 + m2))) - ((G - S*k1*m1)*exp(2*L*m2) - (G + S*k1*m1)*exp(2*L*(m1 + m2)))*(k1*m1**2*sigma2 - k2*m2**2*sigma1) - \
			(2*S*k1*k2*m1*m2**2*sigma1*exp(L*m1) - k1*m1**2*sigma2*(-G + S*k1*m1 + S*k2*m2*exp(2*L*m1) - S*k2*m2 + (G + S*k1*m1)*exp(2*L*m1))*exp(L*m2) + k2*m2**2*(G*sigma1 - Rj*S**2*k1*m1**2)*(exp(2*L*m1) - 1) + \
			(k1*m1**2*sigma2 - k2*m2**2*sigma1)*(-G + S*k1*m1 + (G + S*k1*m1)*exp(2*L*m1)))*exp(2*m2*x_array[i]))*exp(-m2*x_array[i])/(2*S**2*k1*k2*m1**2*m2**2*(-G + S*k1*m1 - S*k2*m2*exp(2*L*m1) + S*k2*m2*exp(2*L*m2) - S*k2*m2*exp(2*L*(m1 + m2)) + \
			S*k2*m2 + (G - S*k1*m1)*exp(2*L*m2) + (G + S*k1*m1)*exp(2*L*m1) - (G + S*k1*m1)*exp(2*L*(m1 + m2))))
		else:
			T2w[i] = 0
	
	T2w0 = T2w[int((N-1)/2)]
	T3w0 = trapz(T2w,x_array)/(2*L)
	return [abs(T2w),abs(T2w0),abs(T3w0),x_array]
	

#Function conversion polynomre T-U thermocouple type S
def polynome_S(T):
	b1 = 5.40313308631
	b2 = 0.012593428974
	b3 = -2.32477968689e-5
	b4=3.22028823036e-8
	b5=-3.31465196389e-11
	b6=2.55744251786e-14
	b7=-1.25068871393e-17
	b8=2.71443176145e-21
	
	U = b1*T + b2*T**2 + b3*T**3 + b4*T**4 + b5*T**5 + b6*T**6 + b7*T**7 + b8*T**8
	return U

#Function conversion polynomre R-T Pt1000
def polynome_pt1000_T_fonction_de_R(R):
	T = (R-1089.6)/(3.85e-3*1089.6)+22.8
	return T
	
#Function conversion polynomre U-T thermocouple type S
def polynome_type_S_T_fonction_de_E(E):
	b12=-5.024805816766898e-43
	b11=3.222219817921142e-38
	b10=-9.108349439714874e-34
	b9=1.493676948877562e-29
	b8=-1.574101642114063e-25
	b7=1.116245567777319e-21
	b6=-5.427735397174568e-18
	b5=1.815268176321127e-14
	b4=-4.160178200931134e-11
	b3=6.588579925468409e-08
	b2=-7.829973449446437e-05
	b1=0.186645256034229
	b0=-0.122638580254693
	
	T = b0 + b1*E + b2*E**2 + b3*E**3 + b4*E**4 + b5*E**5 + b6*E**6 + b7*E**7 + b8*E**8 + b9*E**9 + b10*E**10 + b11*E**11 + b12*E**12
	return T

#function calculate RTD temperature
def RTD_temperature(R,alpha,R0):
	T=(R-R0)/(alpha*R0);
	return T

#function caculate Probe temperature
def Tsonde_static(Usonde,R_ta):
	T_amb = polynome_pt1000_T_fonction_de_R(R_ta)
	U_amb = polynome_S(T_amb)
	U_static = Usonde + U_amb
	T_static = polynome_type_S_T_fonction_de_E(U_static)
	return T_static
	
	
#function calculate contact/noncontact ratio denpend on L and R**2
def contact_ratio(L,R,k1, k2, rho1, rho2, c1, c2, f, sigma1, sigma2, h, Rj, I0, alpha, G, T_amb, N_x):
	M = size(L)
	R_n = empty(M)
	R_a = empty(M)
	R_as = empty(M)
	R_n3 = empty(M)
	R_a3 = empty(M)
	R_as3 = empty(M)
	
	k = (k1 + k2)/2
	rho = (rho1 + rho2)/2
	c = (c1+c2)/2
	sigma = (sigma1 + sigma2)/2
	
	N = 2**6+1 #number of nodes in X
	J = 2**10 #number of nodes in t
	
	for i in range(0,M):
		#if L[i] < 200e-6:
			#N = 2**5+1 #number of nodes in X
			#J = 2**11 #number of nodes in t
		#elif L[i] < 500e-6 and L[i] >= 200e-6:
			#N = 2**6+1 #number of nodes in X
			#J = 2**11 #number of nodes in t
		#else:
			#N = 2**4+1 #number of nodes in X
			#J = 2**11 #number of nodes in t
	
		#calculate values noncontact
		[T,T2w,T_n,T3w_n,x_grid,t_grid] = solve_tc_numerique(k1, k2, rho1, rho2, c1, c2, L[i]*2, R, f, sigma1, sigma2, h, Rj, I0, alpha, 0, T_amb, N,J)
		[T2w_2f,T_a,T3w_a,x_array] = solve_tc_anlytique_2f(k1, k2, rho1, rho2, c1, c2, L[i], R, f, sigma1, sigma2, h, Rj, I0, 0, T_amb,N_x)
		[T2w_1f,T_as,T3w_as,x_array] = solve_tc_anlytique_1f(k, rho, c, L[i], R, f, sigma, h, Rj, I0, 0, T_amb,N_x)
		
		#calculate values contact
		[T,T2w,T_n_c,T3w_n_c,x_grid,t_grid] = solve_tc_numerique(k1, k2, rho1, rho2, c1, c2, L[i]*2, R, f, sigma1, sigma2, h, Rj, I0, alpha, G, T_amb, N,J)
		[T2w_2f,T_a_c,T3w_a_c,x_array] = solve_tc_anlytique_2f(k1, k2, rho1, rho2, c1, c2, L[i], R, f, sigma1, sigma2, h, Rj, I0, G, T_amb,N_x)
		[T2w_1f,T_as_c,T3w_as_c,x_array] = solve_tc_anlytique_1f(k, rho, c, L[i], R, f, sigma, h, Rj, I0, G, T_amb,N_x)
		
		#caculate ratio
		R_n[i] = T_n_c/T_n
		R_a[i] = T_a_c/T_a
		R_as[i] = T_as_c/T_as
		R_n3[i] = T3w_n_c/T3w_n
		R_a3[i] = T3w_a_c/T3w_a
		R_as3[i] = T3w_as_c/T3w_as
	return [R_n,R_a,R_as,R_n3,R_a3,R_as3,N,J]

#Function conversion polynome inverse T-U thermocouple type S

#Funnction read file txt
def readDataFile(filename):
	fhand = open(filename)
	skipline = 0
	for line in fhand:
		line = line.rstrip()
		#skip 'uninteresting lines'
		if not line.startswith('Data'):
			skipline = skipline + 1
		else:
			break
	data=genfromtxt(filename,skip_header=skipline+1)
	return data
	
def readDataFile_2(filename):
	fhand = open(filename)
	skipline = 0
	for line in fhand:
		line = line.rstrip()
		#skip 'uninteresting lines'
		if not line.startswith('Data'):
			skipline = skipline + 1
		else:
			break
	data=genfromtxt(filename,skip_header=skipline+1,comments="%", delimiter=";",filling_values=1)
	return data

import win32clipboard as clipboard

def toClipboardForExcel(array):
    # Create string from array
	line_strings = []
	for line in array:
		line_strings.append("\t".join(line.astype(str)).replace("\n",""))
	array_string = "\r\n".join(line_strings)

	# Put string into clipboard (open, clear, set, close)
	clipboard.OpenClipboard()
	clipboard.EmptyClipboard()
	clipboard.SetClipboardText(array_string)
	clipboard.CloseClipboard()

#data processing, remove unrational points, V2w > 115µV
def remove_highpoint(data, maximum_val, minimum_val):
    data_size = data.shape
    high_point_pos = empty(data_size[0])
    normal_pos = empty(data_size[0])
    normal_data = empty(data_size[0])
    
    b=0
    a=0
    for t in range(0,data_size[0]):
        if data[t]>maximum_val or data[t]<minimum_val :
            high_point_pos[a]=t
            a=a+1
        else:
            normal_pos[b]=t
            normal_data[b]=data[t]
            b=b+1
    
    data_new = empty([b])
    for i in range(0,b):
        data_new[i] = normal_data[i]
        
    return  data_new
	

#calculate average values and errors
def average_matrix(data,nombre_acq_par_point):
    data_size = data.shape
    #nombre_acq_par_point = 5
    a = 0
    if shape(data_size)[0] ==1:
        element = empty(nombre_acq_par_point)
        size_average_matrice = int(data_size[0]/nombre_acq_par_point)
        V2w_m = empty(size_average_matrice)
        EV2w = empty(size_average_matrice)
        for t in range(0,data_size[0],nombre_acq_par_point):
            for i in range(0,nombre_acq_par_point):
                element[i] = data[t+i]  
            V2w_m[a] = mean(element)
            EV2w[a] = std(element)
            a = a+1
    else:
        element = empty([nombre_acq_par_point,data_size[1]])
        size_average_matrice = int(data_size[0]/nombre_acq_par_point)
        V2w_m = empty([size_average_matrice,data_size[1]])
        EV2w = empty([size_average_matrice,data_size[1]])
        for t in range(0,data_size[0],nombre_acq_par_point):
            for i in range(0,nombre_acq_par_point):
                element[i,:] = data[t+i,:]
            V2w_m[a,:] = mean(element, axis=0)
            EV2w[a,:] = std(element, axis=0)
            a = a+1
    result_cal = [V2w_m,EV2w]
    return result_cal
	
#traiter les donnees et sortir résultats et copy to clipboard
def calculs_calibration(data,U0,a,b,K_conv):
	ratio1 = data/(U0-data)
	ratio2 = U0/(U0-data)
	deltaU = (U0-data)
	ks = a/(ratio1 - b)
	Geq = 1/(ratio1*K_conv)
	resultat = vstack((data,deltaU,(1/ratio1),(1/ratio2),data/5.6,Geq,ks)).conj().T
	#toClipboardForExcel(resultat)
	
	return resultat
	
def traitement_data_sphere(data,Rta,alphaRTD,R0_RTD,beta,b0):
	Ta = polynome_pt1000_T_fonction_de_R(Rta)
	T_sonde = Tsonde_static(data[:,8],Rta); data_return = array([T_sonde]).T
	T_RTD= RTD_temperature(data[:,7],alphaRTD,R0_RTD); data_return = insert(data_return, 1, values=T_RTD, axis=1)
	data_return = insert(data_return, 2, values=data[:,4], axis=1) 
	data_return = insert(data_return, 3, values=data[:,9], axis=1)
	P_s = data[:,4] - data[0,4]*(T_RTD-Ta)/(T_RTD[0]-Ta)
	data_return = insert(data_return, 4, values=P_s, axis=1)
	mask = data[:,8] > 2; mask = 1*mask
	G_e = P_s/(T_RTD-T_sonde); data_return = insert(data_return, 5, values=G_e*mask, axis=1)
	G_sonde = P_s/(T_sonde - Ta); data_return = insert(data_return, 6, values=G_sonde*mask, axis=1)
	data_return = insert(data_return, 7, values=data[:,16], axis=1)
	R_c = (T_RTD-Ta)/(data[0,4]*(T_RTD-Ta)/(T_RTD[0]-Ta)); data_return = insert(data_return, 8, values=R_c, axis=1)
	P_c = data[0,4]*(T_RTD-Ta)/(T_RTD[0]-Ta); data_return = insert(data_return, 9, values=P_c, axis=1)
	return data_return
	