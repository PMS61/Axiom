'use client';

import React, { useState, useRef } from 'react';
import CogniBot, { TutorAction } from '@/components/model';
import { GoogleGenAI } from '@google/genai';
import { SUPPORTED_ANIMATIONS, determineAnimation } from '@/lib/tutor-utils';

const DEMO_SCRIPT: TutorAction[] = [
    { type: 'wait', duration: 500 },
    { type: 'animation', name: 'Teacher_Listening', duration: 1000 },
    { type: 'speak', text: 'Welcome. I will be guiding you through this presentation.' },
    { type: 'animation', name: 'Teacher_PointScreen', waitForAction: false },
    { type: 'speak', text: 'As you can see on this slide, the data indicates a significant upward trend.' },
    { type: 'animation', name: 'Teacher_Emphasize', waitForAction: false },
    { type: 'speak', text: 'This is the most critical part of our analysis.' },
    { type: 'animation', name: 'Teacher_SwipeNext', waitForAction: false },
    { type: 'speak', text: 'Let\'s move on to the next topic.' },
    { type: 'animation', name: 'Teacher_ExplainingGestures', waitForAction: false },
    { type: 'speak', text: 'Here we observe the secondary effects of the algorithm.' },
    { type: 'animation', name: 'Teacher_ThinkingPose', waitForAction: false },
    { type: 'speak', text: 'It is fascinating how these variables interact, isn\'t it?' },
    { type: 'animation', name: 'Teacher_Listening', loop: true }
];

export default function TutorDemoPage() {
    const [script, setScript] = useState<TutorAction[]>([]);
    const [currentInput, setCurrentInput] = useState('');
    const [inputText, setInputText] = useState('');
    const [chatHistory, setChatHistory] = useState<{ role: string; content: string }[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const chatContainerRef = useRef<HTMLDivElement>(null);

    // Initialize Gemini API
    const apiKey = process.env.NEXT_PUBLIC_GEMINI_API_KEY;
    const genAI = apiKey ? new GoogleGenAI({ apiKey }) : null;
    const model = genAI ? genAI : null; // The full client object

    // Scroll to bottom of chat when chat history changes
    React.useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [chatHistory, isLoading]);

    const handlePlayDemo = () => {
        setScript(DEMO_SCRIPT);
    };

    const handleGenerateScript = async () => {
        if (!inputText.trim()) return;

        setIsLoading(true);
        // Clear the current script to prevent any conflicts
        setScript([]);

        try {
            // Add user message to chat history
            const updatedChatHistory = [...chatHistory, { role: 'user', content: inputText }];
            setChatHistory(updatedChatHistory);

            if (model) {
                // Create a prompt that guides the AI to respond with both a response and a sequence of actions
                const prompt = `
          You are an AI tutor. Respond to the user's query: "${inputText}"

          Your response should be educational and engaging.

          After your response, suggest a sequence of actions from these types:
          1. Animation actions: ${SUPPORTED_ANIMATIONS.join(', ')}
          2. Speech actions: Include the text to speak

          Here's the meaning of each animation:
          - Teacher_StandingPose: Default idle position
          - Teacher_Listening: When listening to the user
          - Teacher_ExplainingGestures: When explaining concepts with hand movements
          - Teacher_Talking: When delivering information
          - Teacher_ThinkingPose: When considering or reflecting on a question
          - Teacher_PointingBoard: When pointing to a board or wall display
          - Teacher_PointScreen: When pointing to a screen or device
          - Teacher_SwipeNext: When moving to the next topic or slide
          - Teacher_Emphasize: When highlighting important information
          - Teacher_EncouragingNod: When responding positively to correct answers
          - Teacher_LookingAround: When asking for questions or additional input

          Choose the animation and actions that best match the content and tone of your response.

          Format your response as JSON with the following structure:
          {
            "response": "your educational response here",
            "actions": [
              {
                "type": "animation",
                "name": "one of the animation names from the list",
                "duration": optional duration in ms
              },
              {
                "type": "speak",
                "text": "what to say"
              }
            ]
          }

          Only respond with the JSON, nothing else.
        `;

                // Check if the method exists before calling it
                if (!model.models || typeof model.models.generateContent !== 'function') {
                    throw new Error('Model API not properly initialized');
                }

                const response = await model.models.generateContent({
                    model: 'gemini-2.0-flash',  // Use the recommended model
                    contents: prompt
                });

                // Check if response has text property
                if (!response || response.text === undefined) {
                    throw new Error('Invalid response from API');
                }

                const text = response.text;

                let aiResponse = '';
                let actions: TutorAction[] = [];

                try {
                    // Clean the response text to extract JSON
                    const cleanedText = text.replace(/```json\n?|```|json/g, '').trim();

                    // Extract JSON from the response (in case there's extra text)
                    const jsonMatch = cleanedText.match(/\{[\s\S]*\}/);
                    if (jsonMatch) {
                        const parsed = JSON.parse(jsonMatch[0]);
                        aiResponse = parsed.response;

                        // Process the actions array from the AI response
                        if (Array.isArray(parsed.actions)) {
                            for (const action of parsed.actions) {
                                if (action.type === 'animation' && SUPPORTED_ANIMATIONS.includes(action.name)) {
                                    actions.push({
                                        type: 'animation',
                                        name: action.name,
                                        duration: action.duration || 1000,
                                        loop: action.loop !== undefined ? action.loop : false
                                    });
                                } else if (action.type === 'speak' && typeof action.text === 'string') {
                                    actions.push({
                                        type: 'speak',
                                        text: action.text
                                    });
                                } else if (action.type === 'wait' && typeof action.duration === 'number') {
                                    actions.push({
                                        type: 'wait',
                                        duration: action.duration
                                    });
                                }
                            }
                        }

                        // If no actions were generated, create a default sequence
                        if (actions.length === 0) {
                            const defaultAnimation = determineAnimation(aiResponse);
                            actions = [
                                { type: 'animation', name: defaultAnimation, duration: 1000, loop: false },
                                { type: 'speak', text: aiResponse }
                            ];
                        }
                    } else {
                        // Fallback if no JSON found in response
                        console.error('No JSON found in response:', cleanedText);
                        aiResponse = text; // Use the full text as response
                        const defaultAnimation = determineAnimation(aiResponse);
                        actions = [
                            { type: 'animation', name: defaultAnimation, duration: 1000, loop: false },
                            { type: 'speak', text: aiResponse }
                        ];
                    }
                } catch (e) {
                    // Fallback if JSON parsing fails
                    console.error('Failed to parse AI response as JSON:', e);
                    console.error('Response text was:', text);
                    aiResponse = text; // Use the full text as response
                    const defaultAnimation = determineAnimation(aiResponse);
                    actions = [
                        { type: 'animation', name: defaultAnimation, duration: 1000, loop: false },
                        { type: 'speak', text: aiResponse }
                    ];
                }

                // Add AI response to chat history
                const newChatHistory = [...updatedChatHistory, { role: 'model', content: aiResponse }];
                setChatHistory(newChatHistory);

                // Finalize the script with a listening pose at the end
                const finalScript = [...actions, { type: 'animation', name: 'Teacher_Listening', loop: true } as TutorAction];
                setScript(finalScript);
            } else {
                // Fallback without AI integration
                const defaultAnimation = determineAnimation(inputText);
                const fallbackResponse = `I received your input: "${inputText}". To get a proper AI response, please set up your Gemini API key in the environment variables.`;

                // Add to chat history
                const newChatHistory = [...chatHistory, { role: 'user', content: inputText }, { role: 'model', content: fallbackResponse }];
                setChatHistory(newChatHistory);

                const fallbackScript: TutorAction[] = [
                    { type: 'animation', name: defaultAnimation, loop: false, duration: 1000 },
                    { type: 'speak', text: fallbackResponse },
                    { type: 'animation', name: 'Teacher_Listening', loop: true }
                ];
                setScript(fallbackScript);
            }
        } catch (error) {
            console.error('Error calling Gemini API:', error);
            // Fallback to basic response
            const errorMessage = 'I encountered an error processing your request. Could you please try again?';
            const fallbackScript: TutorAction[] = [
                { type: 'animation', name: 'Teacher_ThinkingPose', duration: 1000, loop: false },
                { type: 'speak', text: errorMessage },
                { type: 'animation', name: 'Teacher_Listening', loop: true }
            ];
            setScript(fallbackScript);

            // Also add error to chat history
            const newChatHistory = [...chatHistory, { role: 'user', content: inputText }, { role: 'model', content: errorMessage }];
            setChatHistory(newChatHistory);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (inputText.trim()) {
            await handleGenerateScript();
            setInputText(''); // Clear input after submission
        }
    };

    return (
        <div className="min-h-screen bg-black text-white p-8 font-sans">
            <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8 h-[80vh]">
                {/* Left Panel: Controls and Chat */}
                <div className="lg:col-span-1 space-y-6">
                    <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-purple-500">
                        CogniBot AI Tutor
                    </h1>
                    <p className="text-gray-400">
                        Interactive AI-powered tutoring assistant
                    </p>

                    <div className="bg-white/5 p-6 rounded-2xl border border-white/10 space-y-4">
                        <h2 className="text-xl font-semibold text-cyan-300">Quick Actions</h2>
                        <button
                            onClick={handlePlayDemo}
                            className="w-full py-3 px-4 bg-gradient-to-r from-cyan-600 to-blue-600 rounded-xl font-medium hover:from-cyan-500 hover:to-blue-500 transition-all shadow-lg shadow-cyan-500/20"
                        >
                            ▶ Initialize Demo
                        </button>
                    </div>

                    {/* Chat Interface */}
                    <div className="bg-white/5 p-6 rounded-2xl border border-white/10 space-y-4 flex flex-col h-96">
                        <h2 className="text-xl font-semibold text-purple-300">AI Chat</h2>

                        {/* Chat Messages */}
                        <div
                            ref={chatContainerRef}
                            className="flex-1 overflow-y-auto space-y-3 p-3 bg-black/20 rounded-lg border border-white/10 max-h-64"
                        >
                            {chatHistory.length === 0 ? (
                                <p className="text-gray-400 text-center py-4">Start a conversation with the AI tutor...</p>
                            ) : (
                                chatHistory.map((message, index) => (
                                    <div
                                        key={index}
                                        className={`p-3 rounded-lg max-w-[80%] ${message.role === 'user'
                                                ? 'bg-cyan-900/50 ml-auto text-right'
                                                : 'bg-purple-900/50 mr-auto text-left'
                                            }`}
                                    >
                                        <div className="font-medium text-xs text-gray-400 mb-1">
                                            {message.role === 'user' ? 'You' : 'AI Tutor'}
                                        </div>
                                        <div className="text-sm">{message.content}</div>
                                    </div>
                                ))
                            )}
                            {isLoading && (
                                <div className="p-3 rounded-lg bg-purple-900/50 mr-auto text-left">
                                    <div className="font-medium text-xs text-gray-400 mb-1">AI Tutor</div>
                                    <div className="flex space-x-2">
                                        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"></div>
                                        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce delay-100"></div>
                                        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce delay-200"></div>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Chat Input */}
                        <form onSubmit={handleSubmit} className="flex space-x-2">
                            <input
                                type="text"
                                value={inputText}
                                onChange={(e) => setInputText(e.target.value)}
                                placeholder="Ask the AI tutor anything..."
                                className="flex-1 bg-black/50 border border-white/20 rounded-xl p-3 text-white focus:ring-2 focus:ring-purple-500 outline-none font-mono text-sm"
                                disabled={isLoading}
                            />
                            <button
                                type="submit"
                                disabled={isLoading || !inputText.trim()}
                                className="py-3 px-4 bg-gradient-to-r from-purple-600 to-pink-600 rounded-xl font-medium hover:from-purple-500 hover:to-pink-500 transition-all shadow-lg shadow-purple-500/20 disabled:opacity-50"
                            >
                                Send
                            </button>
                        </form>
                    </div>

                    <div className="text-xs text-gray-500 mt-4">
                        * Requires browser speech synthesis support and Gemini API key.
                    </div>
                </div>

                {/* Right Panel: 3D View */}
                <div className="lg:col-span-2 h-full">
                    <CogniBot
                        key={`cognibot-${script.length}`}  // Re-render when script length changes
                        script={script}
                        autoPlay={true}
                        onComplete={() => console.log('Sequence complete')}
                    />
                </div>
            </div>
        </div>
    );
}
