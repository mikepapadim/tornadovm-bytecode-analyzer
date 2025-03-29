import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.math.TornadoMath;

public class DFT {
    public static void computeDFTFloat(FloatArray inreal, FloatArray inimag,
                                     FloatArray outreal, FloatArray outimag) {
        int n = inreal.getSize();
        for (@Parallel int k = 0; k < n; k++) {
            float sumReal = 0;
            float simImag = 0;
            for (int t = 0; t < n; t++) {
                float angle = (2 * TornadoMath.floatPI() * t * k) / n;
                sumReal += inreal.get(t) * TornadoMath.cos(angle) +
                          inimag.get(t) * TornadoMath.sin(angle);
                simImag += -inreal.get(t) * TornadoMath.sin(angle) +
                          inimag.get(t) * TornadoMath.cos(angle);
            }
            outreal.set(k, sumReal);
            outimag.set(k, simImag);
        }
    }
} 