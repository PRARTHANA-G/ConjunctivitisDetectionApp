
package com.example.conjuntivitis_app
import android.content.Intent
import android.content.Intent.*
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.compose.runtime.Composable
import androidx.compose.ui.tooling.preview.Preview
import com.example.conjuntivitis_app.ml.LiteModel
import com.example.conjuntivitis_app.ui.theme.Conjuntivitis_AppTheme
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer


class MainActivity : ComponentActivity() {
    private lateinit var select_btn: Button
    private lateinit var predict_btn: Button
    private lateinit var camera_btn: Button
    private lateinit var resultView: TextView
    private lateinit var imageView: ImageView
    private lateinit var bitmap: Bitmap
    private lateinit var labels: List<String>
    private lateinit var imageProcessor: ImageProcessor

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // get asset manager
        val assetManager: AssetManager = applicationContext.assets

        // read labels from file
        var labels = assetManager.open("labels.txt").bufferedReader().readLines()

        setContentView(R.layout.activity_main)
        select_btn = findViewById(R.id.select_btn)
        camera_btn = findViewById(R.id.camera_btn)
        predict_btn = findViewById(R.id.predict_btn)
        resultView = findViewById(R.id.resultView)
        imageView = findViewById(R.id.imageView)

        var imageProcessor = ImageProcessor.Builder().add(NormalizeOp(0.0f, 255.0f))
            .add(ResizeOp(120, 120, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        select_btn.setOnClickListener {
            val intent = Intent()
            intent.action = ACTION_GET_CONTENT
            intent.type = "image/*"
            startActivityForResult(intent, 100)
        }

        camera_btn.setOnClickListener {
            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            if (takePictureIntent.resolveActivity(packageManager) != null) {
                val REQUEST_IMAGE_CAPTURE = 1
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
            }
        }

        predict_btn.setOnClickListener {
            var tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            tensorImage = imageProcessor.process(tensorImage)


            val model = LiteModel.newInstance(this)
            val inputFeature0 =
                TensorBuffer.createFixedSize(intArrayOf(1, 120, 120, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(tensorImage.buffer)
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray
            val predictedLabel = if (outputFeature0[0] < 0.5) {
                "white"
            } else {
                "red"
            }
            resultView.text=predictedLabel

/*
            var maxIdx = 0
            outputFeature0.forEachIndexed() { index, fl ->
                if (outputFeature0[maxIdx] < fl) {
                    maxIdx = index
                }
            }
            resultView.text = labels[maxIdx];*/


            model.close()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 100) {
            var uri = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            imageView.setImageBitmap(bitmap)
        } else if (requestCode == 1) {
            try {
                bitmap = data?.extras?.get("data") as Bitmap
                imageView.setImageBitmap(bitmap)
            } catch (e: NullPointerException) {
                e.printStackTrace()
            }
        }
    }
}



fun FloatArray.indexOfMax(): Int {
    var maxIndex = 0
    for (i in indices) {
        if (this[i] > this[maxIndex]) {
            maxIndex = i
        }
    }
    return maxIndex
}



@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    Conjuntivitis_AppTheme {

    }
}
