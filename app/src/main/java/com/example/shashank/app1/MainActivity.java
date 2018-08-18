package com.example.shashank.app1;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.content.res.AssetManager;
import android.graphics.Matrix;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;


//import TensorFlow libraries
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

// Import image utilities
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {


    private TensorFlowInferenceInterface inferenceInterface;
    private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
    /*
    Please note that the name of input and output nodes should match that of names we declared in
    our TensorFlow graph
     */

    private static final String INPUT_NODE = "input"; // our input node
    private static final String OUTPUT_NODE = "output"; // our output node

    private static final int[] INPUT_SIZE = {1,224,224,3};

    static {
        System.loadLibrary("tensorflow_inference");
    }

    // helper function to find the indices of the element in an array with maximum value
    public static int argmax (float [] elems)
    {
        int bestIdx = -1;
        float max = -1000;
        for (int i = 0; i < elems.length; i++) {
            float elem = elems[i];
            if (elem > max) {
                max = elem;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    private int PICK_IMAGE_REQUEST = 1;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

    }
    public void onimg(View view){
        Intent intent = new Intent();
        // Show only images, no videos or anything else
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        // Always show the chooser (if there are multiple options available)
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE_REQUEST);

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {
            inferenceInterface = new TensorFlowInferenceInterface();
            inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);
            System.out.println("model loaded successfully");
            // String imageUri = "drawable://" + R.drawable.models;

            AssetManager assetManager = getAssets();

            Uri uri = data.getData();

            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                // Log.d(TAG, String.valueOf(bitmap));

                ImageView imageView = (ImageView) findViewById(R.id.imageView);
                imageView.setImageBitmap(bitmap);
                final int inputSize=224;

                final int destWidth = 224;
                final int destHeight = 224;
                Bitmap bitmap_scaled = Bitmap.createScaledBitmap(bitmap, destWidth, destHeight, false);
                // Load class names of CIFAR 10 dataset into a string array
                String[] classes = {"airplane","automobile","bird", "cat", "deer", "dog", "frog "," horse", "ship", "truck"};



                int[] intValues = new int[inputSize * inputSize]; // array to copy values from Bitmap image
                float[] floatValues = new float[inputSize * inputSize * 3]; // float array to store image data

                // note: Both intValues and floatValues are flattened arrays

                //get pixel values from bitmap image and store it in intValues
                bitmap_scaled.getPixels(intValues, 0, bitmap_scaled.getWidth(), 0, 0, bitmap_scaled.getWidth(), bitmap_scaled.getHeight());
                for (int i = 0; i < intValues.length; ++i) {
                    final int val = intValues[i];
                /*
                preprocess image if required
                floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
                floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
                floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
                */

                    // convert from 0-255 range to floating point value
                    floatValues[i * 3 + 0] = ((val >> 16) & 0xFF);
                    floatValues[i * 3 + 1] = ((val >> 8) & 0xFF);
                    floatValues[i * 3 + 2] = (val & 0xFF);
                }



                //  the input size node that we declared earlier will be a parameter to reshape the tensor
                // fill the input node with floatValues array
                inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SIZE, floatValues);
                // make the inference
                inferenceInterface.runInference(new String[] {OUTPUT_NODE});
                // create an array filled zeros with dimension of number of output classes. In our case its 10
                float [] result = new float[1010];
                Arrays.fill(result,0.0f);
                // copy the values from output node to the 'result' array
                inferenceInterface.readNodeFloat(OUTPUT_NODE, result);
                // find the class with highest probability
                int class_id=argmax(result);
                TextView textView=(TextView) findViewById(R.id.textView);
                // Setting the class name in the UI
                //System.out.println("classid "+class_id);
                String index=Integer.toString(class_id);
                textView.setText(index);






            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
