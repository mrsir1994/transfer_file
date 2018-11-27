# dictionary that constains id and key positions
kb_d = {'A':(800,485), 'B':(485,695), 'C':(590,275),'D':(800,380),'E':(695,485), 'F':(590,695),
        'G':(695,275), 'H':(590,485), 'I':(590,380),'J':(905,275),'K':(485,275), 'L':(800,590),
        'M':(905,485), 'N':(695,380), 'O':(590,590),'P':(800,695),'Q':(380,275), 'R':(695,590),
        'S':(485,380), 'T':(485,485), 'U':(485,590),'V':(800,275),'W':(380,485), 'X': (905,695),
        'Y':(695,695), 'Z':(380,695), 'DEL':(380,380),'del':(905,590), '_':(380,590),' ':(905,380)}
# dictionary that contains key char and key positions
id_d = {}
for i,key in enumerate(kb_d.keys()):
    kb_d[key] = (i,kb_d[key][0],kb_d[key][1])
    id_d[i] = (key,kb_d[key][0],kb_d[key][1])
   
    
def get_control_vector(current_pos, id_2_kb, prob,q_mouse=1,q_key=1,alpha=3):
    cx,cy = current_pos
    control_vector = np.zeros((2,1))
    for i in range(len(prob)):
        (_,kx,ky) = id_2_kb[i]
        dist = ((cx-kx)**2 + (cy-ky)**2)**0.5
        dir_vec = (np.array([kx-cx, ky-cy])/dist).reshape((2,1))
        
        
        control_vector += dir_vec * alpha * q_mouse * q_key * prob[i]
        
    return control_vector

 