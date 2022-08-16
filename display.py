import pangolin

import numpy as np
import OpenGL.GL as gl

from multiprocessing import Process, Queue

class DisplayMap:
    def __init__(self):
        self.q = Queue()
        self.vp = Process(target=self.viewer_thread, args=(self.q,))
        self.vp.daemon = True
        self.vp.start()

    def viewer_thread(self, q):
        self.viewer_init(1024, 768)
        while True:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        #pangolin.CreateWindowAndBind('MapView', w, h)
        #gl.glEnable(gl.GL_DEPTH_TEST)

        '''self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
            pangolin.ModelViewLookAt(
                0, -10, -8,
                0, 0, 0,
                0, -1, 0
            )
        )
        self.handler = pangolin.Handler3D(self.scam)

        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, w/h)
        self.dcam.SetHandler(self.handler)
        #self.dcam.Resize(pangolin.Viewport(0,0,w*2,h*2))
        self.dcam.Activate(self.scam)'''

        pangolin.CreateWindowAndBind('MapView', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, w, h, w//2, h//2, 0.1, 10000),
            pangolin.ModelViewLookAt(0, -1, -1, 0, 0, 0, pangolin.AxisDirection.AxisY)
        )

        self.handler = pangolin.Handler3D(self.scam)

        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, w/h)
        self.dcam.SetHandler(self.handler)

    def viewer_refresh(self, q):
        while not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.dcam.Activate(self.scam)

        pangolin.glDrawColouredCube()
        pangolin.FinishFrame()



