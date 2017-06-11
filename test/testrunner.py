import os.path
import os

def run(template,tmppath,vars):
    dirname = os.path.split(__file__)[0]
    content = open(os.path.join(dirname,"..","src",template),"r").read()
    gv = vars.copy()
    for g in gv.keys():
        gkey = '%%'+g+'%%'
        if content.find(gkey) >= 0:
            content = content.replace(gkey,str(gv[g]))

    open(tmppath,"w").write(content)
    os.system("python %s -test"%(tmppath))