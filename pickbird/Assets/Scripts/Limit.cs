using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Limit : MonoBehaviour {
    //광고를 보면 DataController의 ad_num이 1씩 오름 --> GetNum, SetNum, AddNum
    //
    // Use this for initialization
    public static bool unlimit;
    public DataController dataController;

    public Button button;
    public Text buttont;
	void Start () {
        unlimit = true;
	}
	
	// Update is called once per frame
	void Update () {
        int limit = dataController.GetNum();
        if(limit > InputController.fakenumber.Count)
        {
            unlimit = true;
        }
        else if(limit == InputController.fakenumber.Count)
        {
            unlimit = false;
        }
        else if(limit < InputController.fakenumber.Count)
        {
            unlimit = false;
        }

        if(unlimit == false)
        {
            
            button.enabled = false;
            if(ButtonController.lang == true)
            {
                buttont.text = "저장불가";
                buttont.fontSize = 80;
            }
           
            if(ButtonController.lang == false)
            {
                buttont.text = "Unable to save";
                buttont.fontSize = 50;
            }
        }
        if(unlimit == true)
        {
           
            button.enabled = true;
            if(ButtonController.lang == true)
            {
                buttont.text = "저장";
                buttont.fontSize = 80;
            }
            
            if(ButtonController.lang ==false )
            {
                buttont.text = "Save";
                buttont.fontSize = 80;
            }


        }
	}
}
