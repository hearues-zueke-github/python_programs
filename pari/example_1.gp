checkUnique(v,n)={
    v2=vector(n);
    unique=1;
    for(i=1,n,
        j=v[i]+1;
        if(v2[j]==1,unique=0;break,v2[j]=1;);
    );
    unique;
};
dc(a,c,m) = {v_=vector(m); x = 0; for (i=1, m-1, v_[i+1] = (a*v_[i]+c)%m;); v_;}
getCycles(m)={
    M=matrix(0,m);
    for(a=0,m-1,
        for(c=0,m-1,
            v_1=dc(a,c,m);
            if(checkUnique(v_1,m),
                M=matconcat([M;v_1]);
            );
        );
    );
    M;
};
getCyclesLength(m)={
    length_=0;
    for(a=0,m-1,
        for(c=0,m-1,
            v_1=dc(a,c,m);
            if(checkUnique(v_1,m),
                length_+=1;
            );
        );
    );
    length_;
};
getCyclesLengths(nn)={
    v=vector(nn);
    for(i=1,nn,
        M=getCycles(i);
        v[i]=matsize(M)[1];
    );
    v;
};
getCyclesLengths2(nn)={
    v=vector(nn);
    for(i=1,nn,
        v[i]=getCyclesLength(i);
    );
    v;
};

s(n)={v=vector(n);for(i=1,n,v[i]=i+1;v[i]+=1;); v;}
sumOwn(v,n)={s=0;for(i=1,n,s+=v[i]); s;}

print(Str("checkUnique([0,1,2],3): ",checkUnique([0,1,2],3)));
print(Str("checkUnique([0,1,1],3): ",checkUnique([0,1,1],3)));

/*position1 = (elt, array) -> for(i = 1, #array, if(array[i] == el, return(i)));*/

getl()={list();}

getv(n)={
    v=vector(n);
    for(i=1,n,
        v[n-i+1]=i;
    );
    v;
}

A=matrix(3, 2)
B=vector(2)
A[1,1] = 4
B[1]=6
B[2]=-2
C=matconcat([B;A;B])
print(Str("A: ", A));
print(Str("B: ", B));
print(Str("C: ", C));

v=getCyclesLengths2(50)
print(Str("v: ", v))
