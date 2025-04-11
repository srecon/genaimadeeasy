package com.blu.genai;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.model.functions.Generator;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.prompt.PromptContext;
import com.github.tjake.jlama.util.Downloader;

import java.io.File;
import java.io.IOException;
import java.util.UUID;

/**
 * Have to use arm based JVM and without SDK manager
 * java 23.0.2 2025-01-21
 * Java(TM) SE Runtime Environment (build 23.0.2+7-58)
 * Java HotSpot(TM) 64-Bit Server VM (build 23.0.2+7-58, mixed mode, sharing)
 * export JDK_JAVA_OPTIONS="--add-modules jdk.incubator.vector --enable-preview"
 * mvn exec:java -Dexec.mainClass=com.blu.genai.App
 */
public class App {
    public static void main(String[] args) throws IOException {
        System.out.println("Testing Jlama!");

        String model = "tjake/Llama-3.2-1B-Instruct-JQ4";
        String workingDirectory = "./models";

        String prompt = "What is the best season to plant avocados?";

        // Downloads the model or just returns the local path if it's already downloaded
        File localModelPath = new Downloader(workingDirectory, model).huggingFaceModel();

        // Loads the quantized model and specified use of quantized memory
        AbstractModel m = ModelSupport.loadModel(localModelPath, DType.F32, DType.I8);

        PromptContext ctx;
        // Checks if the model supports chat prompting and adds prompt in the expected format for this model
        if (m.promptSupport().isPresent()) {
            ctx = m.promptSupport()
                    .get()
                    .builder()
                    .addSystemMessage("You are a helpful chatbot who writes short responses.")
                    .addUserMessage(prompt)
                    .build();
        } else {
            ctx = PromptContext.of(prompt);
        }

        System.out.println("Prompt: " + ctx.getPrompt() + "\n");
        // Generates a response to the prompt and prints it
        // The api allows for streaming or non-streaming responses
        // The response is generated with a temperature of 0.7 and a max token length of 256
        Generator.Response r = m.generate(UUID.randomUUID(), ctx, 0.0f, 256, (s, f) -> {});
        System.out.println(r.responseText);
    }
}
