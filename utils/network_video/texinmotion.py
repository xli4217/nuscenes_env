import numpy as np
import time
import os
import subprocess
import shutil


class TexAnimation:

    def __init__(self,template_file,output_dir = "sequence"):
        self._template_dir,self._template_file = os.path.split(template_file)
        self._output_dir = output_dir
        self._frame_id = 0

        self._silent = True

        self._tikz_styles = {}
        self._newcommands = {}
        self._colors = {}

    def clear_tikz_style(self,style_name):
        self._tikz_styles[style_name] =  {}
        
    def update_tikz_style(self,style_name,attribute_name,new_value):
        if(not style_name in self._tikz_styles.keys()):
            # Insert new element
            self._tikz_styles[style_name] =  {}
        # Update value
        self._tikz_styles[style_name][attribute_name] = new_value

    def update_newcommands(self,command_name,new_value):
        self._newcommands[command_name] = new_value

    def update_color(self,color_name,color_tripple):
        self._colors[color_name] = color_tripple

    def _write_dynamic_file(self):
        with open(os.path.join(self._template_dir,"dynamic.tex"),"w") as f:
            for key, value in self._colors.items():
                f.write("\\definecolor{{{}}}{{rgb}}{{{:0.4f},{:0.4f},{:0.4f}}}\n".format(key,value[0],value[1],value[2]))
            for key, value in self._newcommands.items():
                f.write("\\newcommand{{\\{}}}{{{}}}\n".format(key,value))
            f.write('\\tikzset{%\n')
            for key, style in self._tikz_styles.items():
                f.write("{}/.style={{".format(key))
                first = True
                for attr, value in style.items():
                    # All comma not in the first but all following
                    if(first):
                        first = False
                    else:
                        f.write(",")
                    if(value is None):
                        f.write("{}".format(attr))
                    else:
                        f.write("{}={}".format(attr,value))

                f.write('},\n')
            f.write('}\n')
                

    def render(self):
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        self._write_dynamic_file()

        cmd = ["pdflatex","-shell-escape", "-interaction=nonstopmode", self._template_file]
        self._exec(cmd,cwd = self._template_dir)


        shutil.move(os.path.join(self._template_dir,self._template_file.replace("tex","png")), os.path.join(self._output_dir,"frame_{:05}.png".format(self._frame_id)))
        self._frame_id += 1

    def already_exist(self):
        frame_path = os.path.join(self._output_dir,"frame_{:05}.png".format(self._frame_id))
        return os.path.isfile(frame_path)

    def skip(self):
        self._frame_id += 1

    def to_gif(self,gif_path,delay=2):
        cmd = ["convert","-loop", "0", "-delay", "{}".format(delay), "{}/frame_*.png".format(self._output_dir), "{}".format(gif_path)]
        self._exec(cmd)

    # ffmpeg -framerate 30 -i sequence/frame_%05d.png -c:v libx264 animation.mp4
    def to_mp4(self,mp4_path,frame_rate=30):
        cmd = ["ffmpeg", 
            "-framerate", "{}".format(frame_rate),
            "-i", "{}/frame_%05d.png".format(self._output_dir),
            "-c:v", "libx264", 
            mp4_path
        ]
        self._exec(cmd)


    def _exec(self,cmd,cwd=None):
        pipe_device = None
        if(self._silent):
            pipe_device = subprocess.DEVNULL
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=pipe_device,
            stderr=pipe_device,
            cwd=cwd
        )
        try:
            process.wait(timeout=60*10)
        except:
            process.kill()
            raise ValueError("Process did not terminate within 10 minutes")
        return process.returncode


if __name__ == "__main__":
    t = np.linspace(0,2*np.pi,100)
    sin_seq = 50.0*np.sin(t)+50.0
    cos_seq = 50.0*np.cos(t)+50.0

    frame = TexAnimation("example/template.tex")

    # Loop over each frame
    for i in range(100):

        # Create file with for dynamic variables
        frame.update_newcommands("mycolorsin","green!"+str(int(round(sin_seq[i])))+"!white")
        frame.update_newcommands("mycolorcos","red!"+str(int(round(cos_seq[i])))+"!white")

        frame.render()

    frame.to_gif("animation.gif")