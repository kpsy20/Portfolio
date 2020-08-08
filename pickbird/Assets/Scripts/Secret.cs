using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class Secret : MonoBehaviour {
    public Button button;
    public Text buttont;
	// Use this for initialization
	void Update () {
        if (InputController.enter == true && ButtonController.lang == true)
        { 
            button.enabled = false;
            buttont.text = "준비중";
        }
        if (InputController.enter == true && ButtonController.lang == false)
        {
            button.enabled = false;
            buttont.text = "Preparing";
        }
	}
}
