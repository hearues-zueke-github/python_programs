f(j, acc, va) = {if (j<=0, 0, my(aj=va[j]); (aj + f(j-(acc+aj+1), acc+aj+1, va)) % 10)};
lista(nn) = {va = vector(nn); for (n=1, nn, va[n] = if (n==1, 1, f(n-1, 0, va)); ); va; };
