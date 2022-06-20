for i, pie in enumerate(_pool):
            for j, pie2 in enumerate(_pool[i+1:]):
                flag = 0
                if pie & pie2 is not None:
                    interm = pie | pie2
                    _pool.insert(i+j+1,interm)
                    _pool.pop(i+j+2)
                    flag += 1
                else:
                    pass
            if flag > 0 :
                _pool.pop(0)
            else:
                pass

    def partition(self, return_atoms=None):  # do not return anything but store the final clusters or atoms in clusters into self.clusters
        _pool = deepcopy(self.pool)
        temp1 = _pool
        temp2 = _pool
        self.check = []
        for e in temp1:
            for f in temp2:
                if e & f is not None:
                    f = e | f
                    self.check.append(f)
                else:
                    pass
            if temp1 == temp2:
                break
            else:
                temp1 = temp2
        if temp1 == temp2:pass
        else:print('grammarly wrong, please check it')
        collection = temp1
        self.clusterNumber = len(collection)
        if return_atoms is None: # if return_atoms is True then give the atoms lables in clusters, 
            self.clusters = collection
            print('Please check attribute clusters')
        else:
            series = [np.array(i) for i in collection]
            self.clusters = [self.obj.sortout(j) for j in series]
            print('Please check attribute clusters')
            centroids = np.zeros([self.clusterNumber, 2])
            for k, cluster in enumerate(self.clusters):
                length = len(cluster)
                coords = np.zeros([length, 2])
                for j, a in enumerate(cluster):
                    coords[j,:] = a.coordinate 
                centroids[k,:] = np.mean(coords, axis=0)
            self.centroids = centroids
            return self.centroids