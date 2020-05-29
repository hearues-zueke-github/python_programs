convertToBase(n)={my(l=List(), b=3); while(n>0, listput(l, n%b); n=n\b); l};

digitSum(n, b)={my(dig_sum=0); while(n>0, dig_sum=dig_sum+n%b; n=n\b); dig_sum};

a(n)={my(best_b=1, best_dig_sum=n); if(n>1, for(b=2, n-1, dig_sum=digitSum(n, b); if(best_dig_sum>dig_sum, best_dig_sum=dig_sum; best_b=b))); best_b};
lista(nn)={v=vector(nn); for(i=1, nn, v[i]=a(i)); v};

a2(n)={my(best_b=1, best_dig_sum=n); if(n>1, for(b=2, n-1, dig_sum=sumdigits(n, b); if(best_dig_sum>dig_sum, best_dig_sum=dig_sum; best_b=b))); best_b};
lista2(nn)={v=vector(nn); for(i=1, nn, v[i]=a2(i)); v};
