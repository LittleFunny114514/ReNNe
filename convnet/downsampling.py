from .. import base,cfg,np,misc

class Pooling(base.Operation):
    def __init__(input,self,size:int,mode:str='max'):
        self.size=size
        self.mode=mode
        assert mode in ['max','avg'],'Unsupported pooling mode'
        super().__init__(input)
    
    def choice(self,fm:np.ndarray):
        if self.mode=='max':
            return np.max(fm,axis=(1,2),keepdims=True),\
                np.argmax(fm,axis=(1,2),keepdims=True)
        elif self.mode=='avg':
            return np.sum(fm)/np.prod(fm.shape)

    def forwardUnwrap(self):
        channels=self.input.shape[0]
        fmw=self.input.shape[2]
        fmh=self.input.shape[1]
        if cfg.SCI_BOOST:
            import scipy.ndimage
            if self.mode=='avg':
                self.output=scipy.ndimage.uniform_filter(self.input,size=self.size)
            elif self.mode=='max':
                self.output=scipy.ndimage.maximum_filter(self.input,size=self.size)
        else:
            self.output=np.zeros((channels,fmh//self.size,fmw//self.size),dtype=cfg.dtype)
            for I,J in np.ndindex(self.output.shape[1:]):
                i,j=I*self.size,J*self.size
                self.output[:,I,J],_=self.choice(self.input[:,i:i+self.size,j:j+self.size])
    def backwardUnwrap(self):
        channels=self.input.shape[0]
        fmw=self.input.shape[2]
        fmh=self.input.shape[1]
        self.input=np.zeros((channels,fmh,fmw),dtype=cfg.dtype)
        if self.mode == 'max':
            for I,J in np.ndindex(self.output.shape[1:]):
                i,j=I*self.size,J*self.size
                _,idx=self.choice(self.input[:,i:i+self.size,j:j+self.size])
                self.input[:,i:i+self.size,j:j+self.size][idx]+=self.grad[I,J]