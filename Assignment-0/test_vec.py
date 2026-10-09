from vec import Vec

if __name__ == "__main__":
    v1 = Vec([0, 1, 1.03])
    print(v1) #vector 1
    
    v3 = 2.2 * v1 #rmul
    print(v3)
    
    v3 *= 5 #imul
    print(v3)
    
    v2 = v1 + v3 
    print(v2)

    print(len(v1)) #length of vector 1
    print(v1.norm()) #norm of vector 1
    print(Vec.zeros(3)) #zeroes vector of dimension 3
    print(Vec.ones(3)) #ones vector of dimension 3
    print(Vec.uniform(3)) #uniform vector of dimension 3
    
    print(v1 - v3) #difference of vectors 1 and 3
    print(-v1) #negation of vector 1
    print(1 + v1) #addition of scalar and vector 1
    
    v1 += v3 #addition of vector 3 to vector 1
    print(v1) #result of addition