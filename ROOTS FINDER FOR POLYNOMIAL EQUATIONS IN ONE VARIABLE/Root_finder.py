
def get_root(x,a,ls):
    for i in ls:
        a=a/(x-i)
    a =a**2
    opt = tf.train.AdamOptimizer().minimize(a)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(45000):
        sess.run(opt)
    ls.append( float( sess.run(x) ))
    return ls


import tensorflow as tf
x = tf.Variable([9], dtype=tf.float32)

a = (x**3 - 10*x**2 + 31*x - 30) # = (x-2)(x-3)(x-5)

print("\n\nPlease wait for few seconds.\n\n")

roots =[]

for i in range (3):
    roots = get_root(x,a,roots)

result = [round(i,3) for i in roots]
print("\n\nRoots of the given equation are :" ,result)
