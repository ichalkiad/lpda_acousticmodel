triphones = readdlm("/home/yannis/Desktop/phones.txt")
triphones[:,2]  = int(triphones[:,2])
triphonesToRem = int(readdlm("/home/yannis/Desktop/labels_to_remove.txt"))
triphonesToRem = triphonesToRem .+ 1

labels_drop = triphones[vec(triphonesToRem),:]
