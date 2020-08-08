using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
public class GOScript : MonoBehaviour {
    public Text repick;
    public Text gomenu;
	// Update is called once per frame
	void Update () {
        if (Application.platform == RuntimePlatform.Android)
        {
            if (Input.GetKey(KeyCode.Escape))
            {
                SceneManager.LoadScene("Main");
            }
        }
        if(ButtonController.lang == true)
        {
            repick.text = "다시뽑기";
            gomenu.text = "메뉴로 돌아가기";
        }
        else
        {
            repick.text = "Repick";
            gomenu.text = "Go to Menu";
        }
    }
}
