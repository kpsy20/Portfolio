using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Advertisements;
using UnityEngine.UI;
public class Ads : MonoBehaviour
{
    public DataController dataController;
    
    private const string game_id = "1665235";
    private const string rewarded_video_id = "rewardedVideo";
    // Use this for initialization
    void Start()
    {
        Initialize();
    }
    private void Initialize()
    {
        Advertisement.Initialize(game_id);
    }
    public void ShowRewardedAd()
    {
        if(Advertisement.IsReady(rewarded_video_id))
        {
            var optins = new ShowOptions { resultCallback = HandleShowResult };
            Advertisement.Show(rewarded_video_id, optins);
        }
    }

    private void HandleShowResult(ShowResult result)
    {
        
        switch(result)
        {
            case ShowResult.Finished:
                {
                    
                    
                    dataController.AddNum(1);
                }
                break;
            case ShowResult.Skipped:
                {
                    
                }
                break;
            case ShowResult.Failed:
                {
                    
                }
                break;
        }
    }
}