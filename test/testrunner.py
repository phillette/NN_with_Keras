import os.path
import os

def run(template,tmppath,vars):
    dirname = os.path.split(__file__)[0]

    content_f = open(os.path.join(dirname,"..","src",template),"r")
    content = content_f.read()
    content_f.close()
    gv = vars.copy()
    for g in gv.keys():
        gkey = '%%'+g+'%%'
        if content.find(gkey) >= 0:
            content = content.replace(gkey,str(gv[g]))

    out_f = open(tmppath,"w")
    out_f.write(content)
    out_f.close()
    return 0==os.system("python %s -test"%(tmppath))
