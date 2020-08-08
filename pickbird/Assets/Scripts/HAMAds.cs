using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class HAMAds : MonoBehaviour {
    public DataController dataController;
    public Text numOfAdsDisplayer;
    
    public Ads Ads;
	
	// Update is called once per frame
	void Update () {

        if (ButtonController.lang == true) { numOfAdsDisplayer.text = "현재 조작 가능 인원수: " + dataController.GetNum().ToString(); }
        if(ButtonController.lang == false) { numOfAdsDisplayer.text = "Number of people currently avaiable to manipulate: " + dataController.GetNum().ToString(); }
        
    }
    void Start()
    {
        if(dataController.GetNum() == 0)
        {
            dataController.SetNum(1);
        }
    }
}
