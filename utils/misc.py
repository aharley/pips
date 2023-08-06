import torch
import numpy as np

def posemb_sincos_2d_xy(x, y, temperature=10000, dtype=torch.float32, dim=64):
    device = x.device
    dtype = x.dtype
    B, S = x.shape
    # _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    # y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1) 
    pe = pe.reshape(B,S,dim)
    return pe.type(dtype)

def posemb_sincos_3d(xyz, temperature=10000, dtype=torch.float32, dim=64, cat_coords=False):
    B,N,D = xyz.shape
    assert(D==3)
    device = xyz.device
    dtype = xyz.dtype

    x, y, z = xyz[:,:,0], xyz[:,:,1], xyz[:,:,2]

    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim=1) 
    pe = pe.reshape(B,N,dim*3)

    if cat_coords:
        pe = torch.cat([pe, xyz], dim=2) # B, N, dim*3+3
    
    return pe.type(dtype)

def get_3d_embedding(xyz, C, cat_coords=True):
    B, N, D = xyz.shape
    assert(D==3)

    x = xyz[:,:,0:1]
    y = xyz[:,:,1:2]
    z = xyz[:,:,2:3]
    div_term = (torch.arange(0, C, 2, device=xyz.device, dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C/2))
    
    pe_x = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_z = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    
    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)
    
    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)
    
    pe_z[:, :, 0::2] = torch.sin(z * div_term)
    pe_z[:, :, 1::2] = torch.cos(z * div_term)
    
    pe = torch.cat([pe_x, pe_y, pe_z], dim=2) # B, N, C*3
    if cat_coords:
        pe = torch.cat([pe, xyz], dim=2) # B, N, C*3+3
    return pe
    
class SimplePool():
    def __init__(self, pool_size, version='pt'):
        self.pool_size = pool_size
        self.version = version
        # random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.items = []
        if not (version=='pt' or version=='np'):
            print('version = %s; please choose pt or np')
            assert(False) # please choose pt or np
            
    def __len__(self):
        return len(self.items)
    
    def mean(self, min_size='none'):
        if min_size=='half':
            pool_size_thresh = self.pool_size/2
        else:
            pool_size_thresh = 1
            
        if self.version=='np':
            if len(self.items) >= pool_size_thresh:
                return np.sum(self.items)/float(len(self.items))
            else:
                return np.nan
        if self.version=='pt':
            if len(self.items) >= pool_size_thresh:
                return torch.sum(self.items)/float(len(self.items))
            else:
                return torch.from_numpy(np.nan)
    
    def sample(self):
        idx = np.random.randint(len(self.items))
        return self.items[idx]
    
    def fetch(self, num=None):
        if self.version=='pt':
            item_array = torch.stack(self.items)
        elif self.version=='np':
            item_array = np.stack(self.items)
        if num is not None:
            # there better be some items
            assert(len(self.items) >= num)
                
            # if there are not that many elements just return however many there are
            if len(self.items) < num:
                return item_array
            else:
                idxs = np.random.randint(len(self.items), size=num)
                return item_array[idxs]
        else:
            return item_array
            
    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full
    
    def empty(self):
        self.items = []
        self.num = 0
            
    def update(self, items):
        for item in items:
            if self.num < self.pool_size:
                # the pool is not full, so let's add this in
                self.num = self.num + 1
            else:
                # the pool is full
                # pop from the front
                self.items.pop(0)
            # add to the back
            self.items.append(item)
        return self.items

def farthest_point_sample(xyz, npoint, include_ends=False, deterministic=False):
    """
    Input:
        xyz: pointcloud data, [B, N, C], where C is probably 3
        npoint: number of samples
    Return:
        inds: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    xyz = xyz.float()
    inds = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if deterministic:
        farthest = torch.randint(0, 1, (B,), dtype=torch.long).to(device)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        if include_ends:
            if i==0:
                farthest = 0
            elif i==1:
                farthest = N-1
        inds[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

        if npoint > N:
            # if we need more samples, make them random
            distance += torch.randn_like(distance)
    return inds
    

def farthest_point_sample_py(xyz, npoint):
    N,C = xyz.shape
    inds = np.zeros(npoint, dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N, dtype=np.int32)
    for i in range(npoint):
        inds[i] = farthest
        centroid = xyz[farthest, :].reshape(1,C)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
        if npoint > N:
            # if we need more samples, make them random
            distance += np.random.randn(*distance.shape)
    return inds
    
