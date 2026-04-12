// Supported animation names for CogniBot
export const SUPPORTED_ANIMATIONS = [
    'Teacher_StandingPose',
    'Teacher_Listening',
    'Teacher_ExplainingGestures',
    'Teacher_Talking',
    'Teacher_ThinkingPose',
    'Teacher_PointingBoard',
    'Teacher_PointScreen',
    'Teacher_SwipeNext',
    'Teacher_Emphasize',
    'Teacher_EncouragingNod',
    'Teacher_LookingAround'
];

// Animation mapping based on keywords in the text
export const ANIMATION_KEYWORDS: { [key: string]: string[] } = {
    'Teacher_ExplainingGestures': ['explain', 'describe', 'teach', 'tell me about', 'how does', 'what is', 'concept', 'idea', 'principle', 'rule', 'process', 'method', 'technique', 'strategy'],
    'Teacher_Talking': ['hello', 'hi', 'greetings', 'welcome', 'introduction', 'introduce', 'name', 'pleased to meet'],
    'Teacher_ThinkingPose': ['think', 'consider', 'ponder', 'reflect', 'question', 'doubt', 'wonder', 'analysis', 'analyze', 'evaluation', 'evaluate', 'hmm', 'well', 'interesting'],
    'Teacher_PointingBoard': ['look here', 'see this', 'focus', 'attention', 'notice', 'observe', 'point', 'indicate', 'reference', 'diagram', 'chart', 'graph'],
    'Teacher_PointScreen': ['on screen', 'display', 'show', 'visual', 'image', 'video', 'presentation', 'slide', 'view', 'watch', 'see'],
    'Teacher_SwipeNext': ['next', 'continue', 'proceed', 'advance', 'move on', 'follow up', 'then', 'after', 'subsequently', 'following'],
    'Teacher_Emphasize': ['important', 'crucial', 'essential', 'critical', 'significant', 'key', 'vital', 'focus', 'attention', 'highlight', 'emphasize', 'notice', 'remember', 'keep in mind'],
    'Teacher_EncouragingNod': ['good', 'excellent', 'great', 'well done', 'nice', 'perfect', 'correct', 'right', 'exactly', 'absolutely', 'indeed', 'yes', 'agree', 'understand'],
    'Teacher_LookingAround': ['any questions', 'question', 'ask', 'anything else', 'more', 'further', 'other', 'elaborate', 'detail', 'expand']
};

/**
 * Determines the most appropriate animation based on the text content
 */
export function determineAnimation(text: string): string {
    const lowerText = text.toLowerCase();

    // Check each animation type against its keywords
    for (const [animation, keywords] of Object.entries(ANIMATION_KEYWORDS)) {
        if (keywords.some(keyword => lowerText.includes(keyword))) {
            return animation;
        }
    }

    // Default to listening animation if no keywords match
    return 'Teacher_Listening';
}
