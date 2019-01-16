reverseLst [] = []
reverseLst (x:xs) = reverseLst(xs) ++ [x]

rotL [] = []
rotL (x:xs) = xs ++ [x]

rotR xs = reverseLst(rotL(reverseLst(xs)))

rotAppend1 [] = []
rotAppend1 (x:xs) = [x] ++ rotAppend1(rotL(xs))

fac :: Integer -> Integer
fac 0 = 1
fac 1 = 1
fac x = x * fac(x-1)

main = do  
    -- putStrLn "Hello, what's your name?"  
    -- name <- getLine  
    -- putStrLn ("Hey " ++ name ++ ", you rock!")
    let lst = [1, 2, 3]
    putStrLn(show(lst))

    let lst_rev = reverseLst(lst)
    putStrLn("lst_rev: " ++ show(lst_rev))

    let a = fac(100)
    putStrLn("a: " ++ show(a))

    let lst1 = [i | i <- [0..10]]
    putStrLn("lst1: "++show(lst1))

    let lst1_r = rotR(lst1)
    putStrLn("lst1_r: "++show(lst1_r))
    let lst1_l = rotL(lst1)
    putStrLn("lst1_l: "++show(lst1_l))

    let lst_rot_append1 = rotAppend1(lst1)
    putStrLn("lst_rot_append1: "++show(lst_rot_append1))

    let x = show 123
    putStrLn ("x: " ++ x)
