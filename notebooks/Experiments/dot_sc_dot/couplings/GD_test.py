from typing import Iterable
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt




def base_H(El,Er,B,T,D,Tso,Dso):
    HEl =np.array( [[El+B,0,0,0],[0,El-B,0,0],[0,0,-El-B,0],[0,0,0,-El+B]])
    HEr = np.array( [[Er+B,0,0,0],[0,Er-B,0,0],[0,0,-Er-B,0],[0,0,0,-Er+B]])
    C = np.array([[T,Tso,D,Dso],[Tso,T,-Dso,-D],[D,Dso,-T,-Tso],[-Dso,-D,-Tso,-T]])

    H = np.array([[HEl,C],[C.conj().T,HEr]]).transpose((0,2,1,3)).reshape(8,8)
    return H


base_params = {
"El" :20,
"Er":20,
"B":10,
"T":2,
"D":2,
"Tso": 1,
"Dso": 1}


BL = -base_params['El']*12
BR = base_params['Er']*12
Bc = -2*base_params['El']-base_params['B']
H = base_H(**base_params)






det = np.diag([0,0,0,0,1,1,-1,-1])






def evalf(e,return_vecs =True):
    from typing import Iterable
    if isinstance(e,Iterable):
        res =  np.linalg.eigh(np.stack([H+ee*det for ee in e],axis=0))
    else:
        res = np.linalg.eigh(H+e*det)
    if return_vecs:
        return res
    else:
        return res[0]







def state_selector(target_states,states):
    #compute overlap between target state and states. States are indexed by the first index in both arrays!
    overlaps =np.sum(np.abs(np.einsum('ki,ji->kj',target_states.conj(),states))**2,axis=0)
    #pick the two states with highest overlapwith highest overlap
    return np.argsort(overlaps)[-2:]
    min_Is = [np.argmax(d) for d in overlaps]
    if min_Is[0] == min_Is[1]:
        print(min_Is)
        vals = [overlaps[(i,min_Is[i])] for i in range(2)] # get the corresponding values
        to_sort = overlaps[np.argmin(vals)]
        print(to_sort,vals)
        min_Is = [min_Is[0],np.argsort(to_sort)[-2]]
        print(min_Is)
    
    return min_Is

def energy_select(prev_energies,new_energies):
    # from the new energies, select the two energies that are closes to th two prev_energies
    diffs = np.abs(prev_energies[:,np.newaxis]-new_energies[np.newaxis,:])
    min_Is = [np.argmin(d) for d in diffs]
    if min_Is[0] == min_Is[1]:
        vals = [diffs[(i,min_Is[i])] for i in range(2)]
        to_sort = diffs[np.argmax(vals)]
        min_Is = [min_Is[0],np.argsort(to_sort)[1]]
    
    return min_Is

pl_up =   np.array([1,0,0,0,0,0,0,0])
pr_up =   np.array([0,0,0,0,1,0,0,0])
pr_down = np.array([0,0,0,0,0,1,0,0])
hr_up =   np.array([0,0,0,0,0,0,1,0])
hr_down = np.array([0,0,0,0,0,0,0,1])


T_set = np.stack([pl_up,pr_up])
Tso_set = np.stack([pl_up,pr_down])
D_set = np.stack([pl_up,hr_up])
Dso_set = np.stack([pl_up,hr_down])

sets = [D_set,Tso_set,T_set,Dso_set]





#plt.plot(X,Y)
#plt.plot(X,Y1,ls='--')
#plt.plot(X,Y2,ls='--')


# at xL S1 has higher energy than S2 and at xR S2 has higher energy than S1


def signed_diff(sort_state,states,energies):
    overlaps =np.abs(np.einsum('i,ji->j',sort_state.conj(),states))
    sorting=np.argsort(overlaps)
    return energies[sorting[0]]-energies[sorting[1]]


# get the two states that are the most similar to target states and order them according to their weight relative to S1
# if eneryg dif is positive, we are to the left, if it is negative, we are to the right.



from scipy.optimize import root_scalar

class SignedDiffSearch():
    def __init__(self,xL,xR,evalfunc,xtol,target_states):
            self.xL =xL
            self.xR =xR
            self.f = evalfunc
            self.xtol = xtol
            self.target_states=target_states
    
    def signed_diff(self,x):
        evals,evecs=self.f(x)
        #determine the two relevant states
        relevant_states = state_selector(self.target_states,evecs.T)
        energy_diff = signed_diff(self.target_states[0],np.stack([evecs[:,i] for i in relevant_states]),[evals[i] for i in relevant_states])
        return energy_diff
    def minimize(self):
        res = root_scalar(self.signed_diff,bracket=(self.xL,self.xR),xtol=self.xtol)
        x = res.root
        res['val'] = np.abs(self.signed_diff(x))
        return res


class CleverSignedDiffSearch():
    def __init__(self,xL,xR,evalfunc,xtol,state_sets,method='brentq'):
        self.xL =xL
        self.xR =xR
        self.f = evalfunc
        self.xtol = xtol
        self.target_states=state_sets
        self.__current_i__=None
        self.runs = [[] for _ in range(len(self.target_states))]
        self.calls=0
        self.method = method
        self.initial_brackets = [None]*len(self.target_states)

    def energy_diff(self,overlaps,evals):
        #pick the two states with highest overlapwith highest overlap
        fidelity=np.sum(overlaps,axis=0)
        sort_I=np.argsort(fidelity)
        selection = sort_I[-2:]
        #purity is based the fidelity of the discarded states
        sorting = np.argsort([overlaps[0,s] for s in selection]) # sorted from low to high
        return evals[selection[sorting[1]]]-evals[selection[sorting[0]]],1-0.5*np.sum(fidelity[sort_I[:-2]])
    
    def make_overlap_mat(self,selected_state,other_state,evecs):
        overlaps =np.abs(np.einsum('ki,ji->kj',np.stack([selected_state,other_state]).conj(),evecs))**2
        return overlaps    
    def signed_diff(self,x):
        if isinstance(x,Iterable):
            print(x)
            return [self.signed_diff(xx) for xx in x]
        
        
        evals,evecs=self.f(x)
        #determine the two relevant states
        results = []
        for i,target_states in enumerate(self.target_states):
            
            overlaps = self.make_overlap_mat(target_states[0],target_states[1],evecs.T)
            diff,fidelity=self.energy_diff(overlaps,evals)
            self.runs[i].append((x,diff,fidelity))
            results.append(diff)
        return results[self.__current_i__]
    
    
    @classmethod
    def inspire(cls,old,res_x,percentile=0.5):
        new = cls(old.xL,old.xR,old.f,old.xtol,old.target_states,old.method)

        
        spacing = percentile*(new.xR-new.xL)
        brackets = []
        for i,R in enumerate(old.runs):
            # assume last element is the final guess.
            center=res_x[i]
            print(center)
            #brackets.append((center-spacing,center+spacing))

            N = len(R)
            included = int(np.ceil(N*percentile))
            X_pos = np.array([r[0] for r in R])
            neighbors= np.argsort(np.abs(X_pos-center))[:included]
            # find the two neighbors that are left and rightmost
            neighbor_pos = X_pos[neighbors]
            brackets.append((np.min(neighbor_pos),np.max(neighbor_pos)))
        
        print(brackets)
        new.initial_brackets=brackets
        
        return new
            
        
    
    def find_bracket(self,i,min_fidelity=0.7):
        X = np.array([r[0] for r in self.runs[i] if r[2]>min_fidelity])
        V = np.array([r[1] for r in self.runs[i] if r[2]>min_fidelity])
        L_sign = np.sign(V[np.argmin(X)])
        R_sign = np.sign(V[np.argmax(X)])
        # pick largest x_value with same sign as L_sign:
        X_L= X[np.sign(V)==L_sign]
        V_L= V[np.sign(V)==L_sign]
        X_R= X[np.sign(V)==R_sign]
        V_R= V[np.sign(V)==R_sign]
        if not(len(X_L)) or not(len(X_L)):
            return None
        print(V_L[np.argmax(X_L)],np.max(X_L),V_R[np.argmin(X_R)],np.min(X_R),i)
        
        
        return np.max(X_L),np.min(X_R)
        
    def minimize(self):
        results = []
        bracket =None
        for i in range(len(self.target_states)):
            self.__current_i__ = i
            if self.initial_brackets[i] is None:
                if bracket is None:
                    bracket = (self.xL,self.xR)
                else:
                    bracket=self.find_bracket(i) 
                    if bracket is None: bracket= (self.xL,self.xR) #default if we do not find a bracket
                    #bracket = (self.xL,self.xR)
            else:
                bracket = self.initial_brackets[i] # allows specifying bracket for each run manually
            setup_kwargs = {'method':self.method}
            if self.method == 'halley':
                x0 = 0.5*(bracket[1]+bracket[0] )
                setup_kwargs['x0'] = x0
                setup_kwargs['fprime'] = False
                setup_kwargs['fprime2'] = False
            elif self.method in ('secant','newton'):   
                x0 = 0.5*(bracket[1]+bracket[0] )
                setup_kwargs['x0'] = x0

            try: 
                res = root_scalar(self.signed_diff,xtol=self.xtol,bracket=bracket,**setup_kwargs)
            except Exception as err:
                print('ERROR:',err)
                #fallback to default bracket
                res = root_scalar(self.signed_diff,xtol=self.xtol,bracket=(self.xL,self.xR),**setup_kwargs)
            x = res.root
            res['val'] = np.abs(self.signed_diff(x))
            results.append(res)
            print(bracket,res)
        calls=sum(r.function_calls for r in results)
        print('CALLS:',calls)
        self.calls=calls
        return results   


from matplotlib import pyplot as plt
S=CleverSignedDiffSearch(-100,100,evalf,0.01,sets,method='brentq')
r=S.minimize()
print(r)

#y,v = evalf(r.root)
#ii = state_selector(S.target_states,v.T)
#print(ii)

#for j,yy in enumerate(y):
#    plt.axhline(yy,ls=(':' if j not in ii else '--'))
X = np.linspace(-100,100,100)
Y,V = evalf(X,return_vecs=True)
print(Y.shape,V.shape)
#Is = [state_selector(S.target_states,v.T) for v in V]

#Y2 =np.array([Y[j,i[0]] for j,i in enumerate(Is)])
#Y1 = np.array([Y[j,i[1]] for j,i in enumerate(Is)])
names =('D','Tso','T','Dso')
ls = ('--',':','--',':')
c = ('r','g','b','k')
plt.plot(X,Y)
for _i,rr in enumerate(r):

    plt.axvline(rr.root,label=names[_i],c=c[_i],ls=ls[_i])


print([S.find_bracket(i) for i in range(4)])

    
plt.legend()
plt.show()

_X = np.array([r[0] for r in S.runs[0]])
_X_i = np.argsort(_X)

plt.figure(10)
for i in range(4):
    mask = np.array([r[2] for r in S.runs[i]])[_X_i]
    _Y = (3*i+np.sign([r[1] for r in S.runs[i]]))[_X_i]
    f=plt.plot(_X[_X_i],_Y,ls='--')
    f=plt.scatter(_X[_X_i],_Y,c=mask,ls='--')

plt.colorbar(f)
plt.show()
B=None
sweep = np.linspace(0,5,64)
results = []
total_calls=[]
for i,p in enumerate(names):
    params = base_params.copy()
    p_res = []
    calls=[]
    for s in sweep:
        params[p] = s
        H = base_H(**params)
        sets = (D_set,Tso_set,T_set,Dso_set)
        #pp = []
        #for j in range(len(sets)):
        #    B=SignedDiffSearch(-100,100,evalf,xtol=1e-5,target_states=sets[j])
        #    pp.append(B.minimize())
        if not len(p_res):
            B=CleverSignedDiffSearch(-100,100,evalf,xtol=1e-1,state_sets=sets,method='toms748')
        else:
            B =CleverSignedDiffSearch.inspire(B,[r.root for r in p_res[-1]],percentile=0.25)
        pp=B.minimize()
        p_res.append(pp)
        calls.append(B.calls)
    
    total_calls.extend(calls)
    results.append(p_res)
    
print(np.average(total_calls),np.min(total_calls),np.max(total_calls))
fig,ax=plt.subplots()    
ls = ('-',':','-.','--')
for i,(p,res_list) in enumerate(zip(names,results)):

    ys = np.array([[rr['val']/2 for rr in r] for r in res_list])

    for j,y in enumerate(ys.T):
        if i==j:
            _y = (y-sweep)
            _y[sweep!=0]*=1/sweep[sweep!=0]
        else:
            _y = (y-base_params[names[j]])/base_params[names[j]]
        ax.plot(sweep,_y,label=(p,names[j]),lw=(4-i)*1+1,ls = ls[i])
    #ax.set_yscale('log')
plt.legend()
plt.show()





"""
def state_selector(target_states,states):
    #compute overlap between target state and states. States are indexed by the first index in both arrays!
    overlaps =np.abs(np.einsum('ki,ji->kj',target_states.conj(),states))
    #pick the two states 
    min_Is = [np.argmax(d) for d in overlaps]
    if min_Is[0] == min_Is[1]:
        vals = [overlaps[(i,min_Is[i])] for i in range(2)]
        to_sort = overlaps[np.argmin(vals)]
        min_Is = [min_Is[0],np.argsort(to_sort)[-2]]
    
    return min_Is


def energy_select(prev_energies,new_energies):
    # from the new energies, select the two energies that are closes to th two prev_energies
    diffs = np.abs(prev_energies[:,np.newaxis]-new_energies[np.newaxis,:])
    min_Is = [np.argmin(d) for d in diffs]
    if min_Is[0] == min_Is[1]:
        vals = [diffs[(i,min_Is[i])] for i in range(2)]
        to_sort = diffs[np.argmax(vals)]
        min_Is = [min_Is[0],np.argsort(to_sort)[1]]
    
    return min_Is


from collections import namedtuple

RunInfo = namedtuple('RunInfo',('x','Es','projs','sel'))



class BracketMinimizer():
    def __init__(self,xa,xb,xc,f,target_states,xtol=1e-3,max_iter=100):
        #xa<xb<xc and f(xb)<f(xa), f(xb)<f(xc)
        
        if xb>xc or xa>xb:
            raise ValueError(f'invalid brakcet {xa,xb,xc}')
        
        self.xtol=xtol
        self.target_states = target_states if isinstance(target_states,(list,tuple)) else [target_states,]
        self.f = f
        self.max_iter=max_iter
        self.__current_states__= None

        self.points = []
        # generate the left and the right points automatically:
        for x in (xa,xc):
            res = self.f(x)
            projs = self.make_projections(res[1].T)
            selections = [self.pair_ordering(p)[0] for p in projs]
            self.points.append(RunInfo(x,res[0],projs,selections))
        
        middle_res = self.f(xb)
        middle_proj = self.make_projections(middle_res[1].T)
        res = self.find_selection(xb,middle_res[0],*self.points,middle_proj)
        self.points.append(res)
            
    def make_projections(self,states):
        projs = []
        for ts in self.target_states:
            projs.append(np.abs(np.einsum('aj,bj->ab',ts.conj(),states))**2)#
        return projs
        
    def pair_ordering(self,proj):
        # states indexed by first index!
        projection_fidelity = []
        indices = []
        #generate the list
        # project all the states down to the subspace
        # loop over collumn for all pairs and compute their projection fidelity
        for i in range(proj.shape[1]):
            for j in range(i+1,proj.shape[1]):
                projection_fidelity.append(np.sum(proj[:,i])+np.sum(proj[:,j]))
                indices.append((i,j))
                
        return [indices[i] for i in np.argsort(projection_fidelity)[::-1]] # return pairs sorted accoridng to projection fidelity
    
    
    def find_selection(self,x,energy,left_point,right_point,projs):
        left_delta = x-left_point.x

        right_delta = right_point.x-x
        
        
        
        # pick the first pair of states that give valid differences
        E = energy

        selections = []
        for proj_i,proj in enumerate(projs):
            selection = None
            left_diff = np.abs(left_point.Es[left_point.sel[proj_i][0]]- left_point.Es[left_point.sel[proj_i][1]])
            right_diff = np.abs(right_point.Es[right_point.sel[proj_i][0]]- right_point.Es[right_point.sel[proj_i][1]])
            state_ordering = self.pair_ordering(proj)
            for (i,j) in state_ordering:
                diff = np.abs(E[i]-E[j])
                if (diff-left_diff > left_delta) or  (diff-right_diff>right_delta):
                    continue
                
                selection = (i,j)
                break # 
            if selection is None:
                #raise ValueError('did not find valid pair')
                print('did not find valid pair')
                selection = state_ordering[0]
            selections.append(selection)
        result = RunInfo(x,E,projs,selections) 
        return result
    def update_iter(self,x,res):

        #compute the projections since they will be appended
        projs = self.make_projections(res[1].T)
        
        left_point,right_point = self.find_neighbors(x)
        
        result = self.find_selection(x,res[0],left_point,right_point,projs)
        self.points.append(result)
        
        return result
    
    
    def __call__(self,x):
        
        # compute the eigenvalues and eigenstates:
        res = self.f(x)
        
        ts_i = self.__current_states__ if self.__current_states__ is not None else 0
        #Do the relevant updates
        r=self.update_iter(x,res)
        
        
        relevant_selection = r.sel[ts_i]
        diff = np.abs(r.Es[relevant_selection[0]]-r.Es[relevant_selection[1]]) 
        return diff
    def run(self):
        
        results = []
        old_init = tuple(p for p in self.points)

        run_count = 0
        for ts_i,ts in enumerate(self.target_states):
            #self.points = list(old_init) #reset run
            self.__current_states__ = ts_i
            Bl,Bm,Br = self.find_best_bracket(ts_i)
            print(Bl,Bm,Br)

            from scipy.optimize._optimize import Brent
            
            brent = Brent(self,maxiter=self.max_iter,full_output=True,disp=3,tol=self.xtol)
            
            
            
            f_vals = [np.abs(r.Es[r.sel[ts_i][0]]-r.Es[r.sel[ts_i][1]]) for r in (Bl,Bm,Br)]
            
            def override_bracket_constructor(*args):
                return (Bl.x,Bm.x,Br.x,*f_vals,0) # 0 is func_calls
            
            brent.get_bracket_info = override_bracket_constructor
            
            brent.optimize()
            run_count += brent.funcalls
            res = (brent.xmin,brent.fval) 
            results.append(res)
        print(run_count)
        return results

    def find_neighbors(self,x):
        xvals = np.array([r.x for r in self.points])
        is_left = np.where(xvals<x)
        closest_left = is_left[0][np.argmax(xvals[is_left])]
        is_right = np.where(xvals>x)
        closest_right = is_right[0][np.argmin(xvals[is_right])]
        
        return self.points[closest_left],self.points[closest_right]

    def find_best_bracket(self,ts_i):
        # find lowest three energy values (assuming convexity!) and arange in the bracket 
        
        #sort the points according to the y and fixing the largest one, find the two lower ones such that 
        #x0<x1<x2 and f(x1) smaller than the other two 
        
        
        
        ys = [np.abs(r.Es[r.sel[ts_i][0]]-r.Es[r.sel[ts_i][1]]) for r in self.points]
        xs = [r.x for r in self.points]
        lowest = np.argsort(ys)
        for _i in range(2,len(lowest)):
            i = lowest[_i]
            for _j in range(1,_i):
                j = lowest[_j]
                for _k in range(_j):
                    k=lowest[_k]
                    if max(xs[i],xs[j],xs[k])-min(xs[i],xs[j],xs[k])<self.xtol*2:
                        continue # don't make bracket too small
                    if xs[i]>max(xs[j],xs[k]):
                        # i is right-most
                        if (xs[j]<xs[k]) and (ys[k]<ys[j]):
                            return self.points[j],self.points[k],self.points[i]
                        elif (xs[j]>xs[k]) and (ys[j]<ys[k]):
                            return self.points[k],self.points[j],self.points[i]
                    elif xs[i]<min(xs[j],xs[k]):
                        if (xs[j]<xs[k]) and (ys[j]<ys[k]):
                            return self.points[i],self.points[j],self.points[k]
                        elif (xs[j]>xs[k]) and (ys[j]>ys[k]):
                            return self.points[i],self.points[k],self.points[j]
        raise ValueError('invalid brackets')



pl_up =   np.array([1,0,0,0,0,0,0,0])
pr_up =   np.array([0,0,0,0,1,0,0,0])
pr_down = np.array([0,0,0,0,0,1,0,0])
hr_up =   np.array([0,0,0,0,0,0,1,0])
hr_down = np.array([0,0,0,0,0,0,0,1])


T_set = np.stack([pl_up,pr_up])
Tso_set = np.stack([pl_up,pr_down])
D_set = np.stack([pl_up,hr_up])
Dso_set = np.stack([pl_up,hr_down])

sets = [T_set,Tso_set,D_set,Dso_set]

    
names =('T','Tso','D','Dso')
        
e_vals= np.linspace(BL,BR,256)

evals = evalf(e_vals,return_vecs=True)
plt.plot(e_vals,evals[0],ls=':')
this_set = Dso_set
#func_inst = func(this_set)
B=BracketMinimizer(BL,Bc,BR,evalf,sets)
res = B.run()

#diff = [func_inst(e) for e in e_vals]

#plt.plot(e_vals,diff,'ro')
#S = np.stack(func_inst.selections)


#func_inst = func((pl_down,hr_down))
#diff2 = [func_inst(e) for e in e_vals[::-1]]
#S2 = np.stack(func_inst.selections)
#plt.plot(e_vals,diff2[::-1],'bo')
#plt.plot(e_vals,S2[::-1],ls='--')
#plt.plot(e_vals,S[::],)
EEs = []
for i,_ in enumerate(e_vals):
    E=evals[0][i]
    psi = evals[1][i]
    selection = state_selector(this_set,psi.T)
    EEs.append([E[selection[0]],E[selection[1]]])

EEs = np.sort(EEs,axis=1)
plt.plot(e_vals,EEs)
colors = ('red','green','orange','blue')
for r,name,c in zip(res,names,colors):
    plt.axvline(r[0],label=(name,r[1]/2),c=c)

plt.legend()
#_X = [b.x for b in B.points]
#_Ys = np.stack([b[1] for b in B.hist])

#plt.plot(_X,_Ys[:,0],ls='--')
#plt.plot(_X,_Ys[:,1],ls='--')

plt.show()



sweep = np.linspace(0,5,64)
results = []
for i,p in enumerate(names):
    params = base_params.copy()
    p_res = []
    for s in sweep:
        params[p] = s
        H = base_H(**params)
        sets = (T_set,Tso_set,D_set,Dso_set)
        B=BracketMinimizer(BL,Bc,BR,evalf,sets,xtol=1e-5)
        p_res.append(B.run())
    results.append(p_res)
    
fig,ax=plt.subplots()    
ls = ('-',':','-.','--')
for i,(p,res_list) in enumerate(zip(names,results)):

    ys = np.array([[rr[1]/2 for rr in r] for r in res_list])
    for j,y in enumerate(ys.T):
        if i==j:
            _y = sweep-y
        else:
            _y = y-base_params[names[j]]
        ax.plot(sweep,_y,label=(p,names[j]),lw=(4-i)*1+1,ls = ls[i])
    #ax.set_yscale('log')
plt.legend()
plt.show()





class UniSweep():
    def __init__(self,x_min,x_max,)
    
"""